"""Core regression tests for the Uh-guided Udeco experiment."""

from pathlib import Path

import numpy as np

from udeco_uh_roi_edge_detection import (
    ExperimentConfig,
    baseline_sigmas,
    build_direct_event_table,
    detect_roi_candidates,
    find_uh_rois,
    estimate_motion_frequency,
    load_paired_sample,
    localize_roi_candidates,
    resolve_motion_interval,
)


def test_paired_loader_keeps_uh_and_udeco_on_one_repaired_time_axis():
    path = Path("subject_data/cgx/cgx_rope_1_sensor.csv")
    sample = load_paired_sample(path, data_root=Path("subject_data"), fs=100)
    assert set(sample["signals"]) == {"Uh2", "Udeco2", "Uh3", "Udeco3"}
    assert all(len(values) == len(sample["time_s"]) for values in sample["signals"].values())
    assert len(sample["time_s"]) == 20860
    assert np.isfinite(sample["signals"]["Uh2"]).all()


def test_baseline_sigmas_ignore_events_after_first_fifteen_seconds():
    rng = np.random.default_rng(7)
    uh = rng.normal(0, 0.1, 200)
    udeco = rng.normal(0, 0.01, 200)
    first = baseline_sigmas(uh, udeco, fs=10, baseline_sec=15.0)
    uh[160:] += 100
    udeco[160:] += 100
    second = baseline_sigmas(uh, udeco, fs=10, baseline_sec=15.0)
    assert first == second
    assert np.isclose(first["sigma_du"], np.std(np.diff(udeco[:150])))


def test_distance_selects_main_peaks_without_transitive_roi_merge():
    x = np.zeros(300)
    x[100], x[120], x[140], x[160], x[180] = 10, 4, -2, 4, 11
    rois, peaks = find_uh_rois(
        x, fs=100, sigma_uh=0.1, sigma_duh=0.1,
        motion_interval={"start_idx": 0, "end_idx": 300}, f0_hz=1.0,
        config=ExperimentConfig())
    assert [peak["peak_idx"] for peak in peaks] == [100, 120, 160, 180]
    assert [peak["peak_idx"] for peak in peaks if peak["is_main"]] == [100, 180]
    assert [roi["main_peak_idx"] for roi in rois] == [100, 180]
    assert rois[0]["right_idx"] == 140
    assert rois[1]["left_idx"] == 141
    assert [peak["roi_id"] for peak in peaks] == ["ROI_0", "ROI_0", "ROI_1", "ROI_1"]


def test_uh_peak_must_exceed_mean_of_both_coarse_bases():
    signal = np.full(30, 8.0)
    signal[9] = 0.0
    signal[15] = 10.0
    config = ExperimentConfig(uh_peak_prominence_sigma=0.5,
                              uh_peak_to_coarse_mean_sigma=7.0)

    rois, peaks = find_uh_rois(signal, fs=100, sigma_uh=1.0,
                               sigma_duh=1.0,
                               motion_interval={"start_idx": 0, "end_idx": 30},
                               f0_hz=1.0, config=config)
    assert peaks == rois == []

    _, shifted_peaks = find_uh_rois(signal + 100.0, fs=100, sigma_uh=1.0,
                                   sigma_duh=1.0,
                                   motion_interval={"start_idx": 0, "end_idx": 30},
                                   f0_hz=1.0,
                                   config=ExperimentConfig(uh_peak_prominence_sigma=0.5,
                                                           uh_peak_to_coarse_mean_sigma=5.0))
    assert [peak["peak_idx"] for peak in shifted_peaks] == [15]
    assert shifted_peaks[0]["coarse_mean_rise"] == 6.0

    _, stricter_peaks = find_uh_rois(signal, fs=100, sigma_uh=1.0,
                                    sigma_duh=1.0,
                                    motion_interval={"start_idx": 0, "end_idx": 30},
                                    f0_hz=1.0,
                                    config=ExperimentConfig(uh_peak_prominence_sigma=0.5,
                                                            uh_peak_to_coarse_mean_sigma=6.0))
    assert stricter_peaks == []


def test_motion_interval_and_fft_are_channel_local_and_motion_only():
    fs = 100
    time = np.arange(4000) / fs
    signal = np.sin(2 * np.pi * 2 * time)
    motion = {"start_idx": 1000, "end_idx": 3000}
    spectrum = estimate_motion_frequency(signal, fs=fs, motion_interval=motion)
    assert abs(spectrum["f0_hz"] - 2.0) < 0.05
    assert estimate_motion_frequency(signal, fs=fs, motion_interval=motion,
                                     override_hz=1.5)["f0_hz"] == 1.5
    sample = {"fs": fs, "time_s": time}
    assert resolve_motion_interval(sample, baseline_sec=15.0,
                                   manual_start_sec=10, manual_end_sec=30)["end_idx"] == 3000
    spikes = np.zeros(4000)
    spikes[[500, 1100, 1120, 1200, 3500]] = 3
    rois, peaks = find_uh_rois(spikes, fs=fs, sigma_uh=0.1, sigma_duh=0.1,
                               motion_interval=motion, f0_hz=2.0, config=ExperimentConfig())
    assert [peak["peak_idx"] for peak in peaks] == [1100, 1120, 1200]
    assert len(rois) == 2
    assert rois[0]["main_peak_idx"] in (1100, 1120)
    assert rois[1]["main_peak_idx"] == 1200


def test_main_peak_distance_equality_and_long_gap():
    values = np.zeros(500)
    values[100], values[175], values[300], values[475] = 5, 6, 5, 6
    config = ExperimentConfig()
    rois, peaks = find_uh_rois(
        values, fs=100, sigma_uh=0.1, sigma_duh=0.1,
        motion_interval={"start_idx": 0, "end_idx": 500}, f0_hz=1.0,
        config=config)
    assert [roi["main_peak_idx"] for roi in rois] == [100, 175, 300, 475]
    assert rois[0]["right_idx"] < rois[1]["left_idx"]
    assert rois[2]["right_idx"] < rois[3]["left_idx"]
    assert rois[2]["right_idx"] < 400 < rois[3]["left_idx"]

    values[175] = 0
    values[174] = 6
    rois, _ = find_uh_rois(
        values, fs=100, sigma_uh=0.1, sigma_duh=0.1,
        motion_interval={"start_idx": 0, "end_idx": 500}, f0_hz=1.0,
        config=config)
    assert [roi["main_peak_idx"] for roi in rois if roi["main_peak_idx"] < 200] == [174]


def test_udeco_detection_extends_only_the_left_side_of_logical_roi():
    signal = np.zeros(2500)
    signal[1798:1820] = 0.01
    roi = {"roi_id": "ROI_0", "left_idx": 1800, "right_idx": 1850,
           "search_left_idx": 1797, "main_peak_idx": 1810}
    candidates, _ = detect_roi_candidates(signal, [roi], fs=100,
                                          config=ExperimentConfig())
    assert any(candidate["direction"] == 1 and candidate["center_idx"] < 1800
               for candidate in candidates)
    assert all(1797 <= candidate["center_idx"] <= 1850 for candidate in candidates)


def test_derivative_extrema_selects_one_signed_boundary_per_roi():
    rng = np.random.default_rng(23)
    signal = np.zeros(2500)
    signal[:1500] = rng.normal(0, 1e-4, 1500)
    signal[1798:1820] = 0.01
    signal[1851:] = 0.05  # Larger edge just outside the ROI must be ignored.
    roi = {"roi_id": "ROI_0", "left_idx": 1800, "search_left_idx": 1797,
           "right_idx": 1850, "main_peak_idx": 1810}

    candidates, diagnostics = detect_roi_candidates(
        signal, [roi], fs=100,
        config=ExperimentConfig())

    assert [(row["center_idx"], row["direction"]) for row in candidates] == [
        (1798, 1), (1820, -1)]
    assert all(row["region_start_idx"] == row["center_idx"] == row["region_end_idx"]
               for row in candidates)
    assert all(row["detector_name"] == "derivative_extrema" for row in candidates)
    sigma_du = np.std(np.diff(signal[:1500]))
    assert np.isclose(diagnostics["derivative_sigma"], sigma_du)
    assert np.isclose(candidates[0]["detector_score"], 0.01 / sigma_du)


def test_below_gate_extremum_remains_available_for_diagnostics():
    rng = np.random.default_rng(24)
    signal = np.zeros(2500)
    signal[:1500] = rng.normal(0, 1e-4, 1500)
    signal[1800:1860] = np.arange(60) * 1e-5
    roi = {"roi_id": "ROI_0", "left_idx": 1801, "right_idx": 1850,
           "main_peak_idx": 1820}

    candidates, _ = detect_roi_candidates(
        signal, [roi], fs=100,
        config=ExperimentConfig())

    assert len(candidates) == 1
    assert candidates[0]["direction"] == 1
    assert candidates[0]["detector_score"] < 3.0


def test_extrema_entry_uses_strict_five_sigma_and_either_direction():
    from udeco_uh_roi_edge_detection import recover_missing_extrema_pairs

    noise = {"sigma_udeco": .5, "sigma_du": 1.0}
    roi = {"roi_id": "ROI_0", "left_idx": 118, "search_left_idx": 117,
           "right_idx": 130, "main_peak_idx": 122}
    for peak, enters in ((5.0, False), (6.0, True)):
        signal = np.zeros(160)
        signal[:11] = [0, 1] * 5 + [0]  # Ten alternating differences: sigma_du = 1.
        signal[120:125] = peak
        signal[125:] = peak - 3  # Only the rising extremum can exceed 5 sigma.
        candidates, _ = detect_roi_candidates(
            signal, [roi], fs=100,
            config=ExperimentConfig(baseline_sec=.11))
        rows = localize_roi_candidates(
            signal, signal, candidates, [roi], fs=100, noise=noise,
            config=ExperimentConfig())
        assert all(row.get("extrema_entry_pass") is enters for row in rows)
        if not enters:
            assert all(row["reason"] == "below_extrema_entry_threshold" for row in rows)
            recovered = recover_missing_extrema_pairs(
                signal, rows, [roi], fs=100, noise=noise,
                config=ExperimentConfig())
            assert any(row["detector_name"] == "peak_fallback" for row in recovered)
        else:
            assert any(row.get("right_mode") for row in rows)
            assert [row["direction"] for row in candidates] == [1]


def _five_sigma_region_signal(changes):
    signal = np.zeros(180)
    signal[:11] = [0, 1] * 5 + [0]  # Baseline sigma_du = 1.
    increments = np.zeros(80)
    for center, value in changes.items():
        increments[center - 100] = value
    signal[100:] = np.cumsum(increments)
    return signal


def test_nearest_uh_peak_region_wins_independently_for_each_direction():
    signal = _five_sigma_region_signal({106: 20, 118: 7, 123: -8, 134: -20})
    roi = {"roi_id": "ROI_0", "left_idx": 100, "search_left_idx": 100,
           "right_idx": 140, "main_peak_idx": 120}
    candidates, diagnostics = detect_roi_candidates(
        signal, [roi], fs=100, config=ExperimentConfig(baseline_sec=.11))
    assert [(row["center_idx"], row["direction"]) for row in candidates] == [
        (118, 1), (123, -1)]
    assert [(row["region_start_idx"], row["region_end_idx"]) for row in candidates] == [
        (118, 118), (123, 123)]
    regions = diagnostics["regions"]
    assert sum(row["direction"] == 1 for row in regions) == 2
    assert sum(row["direction"] == -1 for row in regions) == 2
    assert {row["center_idx"] for row in regions if row["selected"]} == {118, 123}


def test_region_single_gap_can_reverse_sign_but_second_gap_splits_it():
    signal = _five_sigma_region_signal(
        {110: 8, 111: 9, 112: -8, 113: 7, 116: 6})
    roi = {"roi_id": "ROI_0", "left_idx": 100, "right_idx": 140,
           "main_peak_idx": 111}
    candidates, diagnostics = detect_roi_candidates(
        signal, [roi], fs=100, config=ExperimentConfig(baseline_sec=.11))
    positives = [row for row in diagnostics["regions"] if row["direction"] == 1]
    assert [(row["region_start_idx"], row["region_end_idx"]) for row in positives] == [
        (110, 113), (116, 116)]
    chosen = next(row for row in candidates if row["direction"] == 1)
    assert (chosen["center_idx"], chosen["region_start_idx"], chosen["region_end_idx"]) == (
        111, 110, 113)

    signal = _five_sigma_region_signal({110: 8, 111: -1, 112: 8,
                                        113: 0, 114: 9})
    _, diagnostics = detect_roi_candidates(
        signal, [roi], fs=100, config=ExperimentConfig(baseline_sec=.11))
    positives = [row for row in diagnostics["regions"] if row["direction"] == 1]
    assert [(row["region_start_idx"], row["region_end_idx"]) for row in positives] == [
        (110, 112), (114, 114)]


def test_region_tie_uses_stronger_peak_then_earlier_time_and_missing_sign_falls_back():
    from udeco_uh_roi_edge_detection import recover_missing_extrema_pairs

    roi = {"roi_id": "ROI_0", "left_idx": 100, "right_idx": 140,
           "main_peak_idx": 120}
    signal = _five_sigma_region_signal({119: 7, 121: 8})
    candidates, _ = detect_roi_candidates(
        signal, [roi], fs=100, config=ExperimentConfig(baseline_sec=.11))
    assert [(row["center_idx"], row["direction"]) for row in candidates] == [(121, 1)]
    localized = localize_roi_candidates(
        signal, signal, candidates, [roi], fs=100,
        noise={"sigma_udeco": .5, "sigma_du": 1}, config=ExperimentConfig())
    recovered = recover_missing_extrema_pairs(
        signal, localized, [roi], fs=100,
        noise={"sigma_udeco": .5, "sigma_du": 1}, config=ExperimentConfig())
    assert sum(row["detector_name"] == "peak_fallback" for row in recovered) == 2

    signal = _five_sigma_region_signal({119: 8, 121: 8})
    candidates, _ = detect_roi_candidates(
        signal, [roi], fs=100, config=ExperimentConfig(baseline_sec=.11))
    assert candidates[0]["center_idx"] == 119


def test_extrema_edge_can_end_far_beyond_roi_search_boundary():
    steps = np.zeros(2200)
    steps[1846:1905] = .0005
    steps[1848] = .0006
    signal = np.cumsum(steps)
    roi = {"roi_id": "ROI_0", "left_idx": 1800,
           "search_left_idx": 1797, "right_idx": 1850,
           "main_peak_idx": 1848}
    candidates, _ = detect_roi_candidates(
        signal, [roi], fs=100,
        config=ExperimentConfig())
    rising = next(row for row in candidates if row["direction"] == 1)
    measured = localize_roi_candidates(
        signal, signal, [rising], [roi], fs=100,
        noise={"sigma_udeco": 1e-4, "sigma_du": 1e-4},
        config=ExperimentConfig())[0]
    assert rising["center_idx"] == 1848
    assert measured["resolved"]
    assert measured["start_idx"] < 1849
    assert measured["end_idx"] > roi["right_idx"] + 20
    assert measured["right_mode"] == "positive_general"


def _direct_edge(roi_id, center, direction, *, delta=None, resolved=True, is_sharp=False):
    row = {"roi_id": roi_id, "center_idx": center, "direction": direction,
           "candidate_direction": direction,
           "region_start_idx": center, "region_end_idx": center,
           "detector_name": "derivative_extrema", "detector_score": 8.0,
           "resolved": resolved, "reason": None if resolved else "core_does_not_end",
           "is_sharp": is_sharp}
    if resolved:
        row.update(start_idx=center - 1, end_idx=center,
                   delta=abs(delta), T_edge=0.01,
                   derivative_peak_width=0.01, normalized_derivative_peak=8.0)
    return row


def _direct_table(localized, rois):
    return build_direct_event_table(
        localized, rois, channel="Udeco2", sigma_udeco=1e-4, fs=100,
        config=ExperimentConfig(),
        direction_map={1: {"sharp": "detach", "gradual": "release"},
                       -1: {"sharp": "attach", "gradual": "force"}})


def test_direct_selection_displays_both_extrema_without_position_pair_gate():
    rois = [{"roi_id": "ROI_0", "left_idx": 100, "search_left_idx": 97,
             "main_peak_idx": 160, "right_idx": 220}]
    table = _direct_table([
        _direct_edge("ROI_0", 105, 1, delta=0.01),
        _direct_edge("ROI_0", 180, -1, resolved=False)], rois)
    assert table.is_main_rising.tolist() == [True, False]
    assert table.is_main_falling.tolist() == [False, True]
    assert table.pair_status.tolist() == ["unpaired", "unpaired"]
    assert table.event_label.tolist() == [None, None]
    assert table.classification_status.tolist() == ["roi_missing_pair", "roi_missing_pair"]


def test_direct_classification_uses_strict_amplitude_and_flags_direction_conflict():
    rois = [{"roi_id": "ROI_0", "left_idx": 100, "right_idx": 200},
            {"roi_id": "ROI_1", "left_idx": 201, "right_idx": 300}]
    first = _direct_edge("ROI_0", 150, 1, delta=0.0002)
    second = _direct_edge("ROI_1", 250, 1, delta=0.0003)
    second["candidate_direction"] = -1
    table = _direct_table([first, second], rois)
    assert table.pair_status.tolist() == ["unpaired", "unpaired"]
    assert table.event_label.tolist() == [None, None]
    assert table.classification_status.tolist() == ["roi_missing_pair", "roi_missing_pair"]
    assert table.direction_delta_conflict.tolist() == [False, True]


def test_direct_cross_roi_duplicate_is_diagnostic_only():
    rois = [{"roi_id": "ROI_0", "left_idx": 100, "right_idx": 200},
            {"roi_id": "ROI_1", "left_idx": 201, "search_left_idx": 198,
             "right_idx": 300}]
    table = _direct_table([
        _direct_edge("ROI_0", 199, 1, delta=0.01),
        _direct_edge("ROI_1", 199, 1, delta=0.01)], rois)
    assert table.overlap_duplicate.tolist() == [True, False]
    assert table.is_main_rising.tolist() == [True, True]
    assert table.event_label.tolist() == [None, None]


def test_single_slow_uh_peak_uses_refined_bounds_without_margin():
    uh = np.zeros(300)
    uh[100:151] = np.linspace(0, 5, 51)
    uh[151:202] = np.linspace(4.9, 0, 51)
    rois, peaks = find_uh_rois(uh, fs=100, sigma_uh=0.1, sigma_duh=0.1,
                           motion_interval={"start_idx": 0, "end_idx": 300}, f0_hz=1.0,
                           config=ExperimentConfig())
    assert len(rois) == 1
    assert rois[0]["left_idx"] == peaks[0]["refined_left_idx"]
    assert rois[0]["right_idx"] == peaks[0]["refined_right_idx"]


def test_extrema_localization_keeps_roi_anchor_when_stronger_peak_is_outside():
    signal = np.zeros(2200)
    signal[1800:] = .01
    signal[1810:] += .02
    roi = {"roi_id": "ROI_0", "left_idx": 1798, "search_left_idx": 1797,
           "right_idx": 1801, "main_peak_idx": 1800}
    cfg = ExperimentConfig()
    candidates, _ = detect_roi_candidates(signal, [roi], fs=100, config=cfg)
    measured = localize_roi_candidates(signal, signal, candidates, [roi], fs=100,
                                       noise={"sigma_udeco": 1e-4, "sigma_du": 1e-4}, config=cfg)[0]
    assert measured["derivative_peak_idx"] == 1800
    assert np.isclose(measured["core_threshold"], .0015)


def test_overlapping_extrema_split_at_quietest_derivative_and_remeasure():
    from udeco_uh_roi_edge_detection import _split_extrema_overlaps
    signal = np.zeros(100)
    signal[20:41] = 1
    signal[41:] = .2
    rows = [dict(roi_id="ROI_0", detector_name="derivative_extrema", center_idx=20,
                 direction=1, start_idx=19, end_idx=45, resolved=True),
            dict(roi_id="ROI_0", detector_name="derivative_extrema", center_idx=41,
                 direction=-1, start_idx=35, end_idx=42, resolved=True)]
    _split_extrema_overlaps(rows, signal, fs=100,
                            noise={"sigma_udeco": .01, "sigma_du": .001}, config=ExperimentConfig())
    first, second = rows
    assert first["overlap_split_idx"] == second["overlap_split_idx"] == 21
    assert first["end_idx"] == second["start_idx"] == 21
    assert first["end_rule"] == "overlap_split"
    assert first["T_edge"] == .02
    assert first["delta"] == 1
    assert np.isclose(second["delta"], .8)
    assert second["direction"] == -1


def test_zero_length_overlap_edges_retrace_to_third_reversal():
    from udeco_uh_roi_edge_detection import _split_extrema_overlaps

    signal = np.zeros(40)
    signal[16:20] = [.3, .2, .1, 0]
    signal[20:24] = 1
    signal[24:26] = [.2, .3]
    signal[26:] = .4
    signal[27:] = .5
    rows = [dict(roi_id="ROI_0", detector_name="derivative_extrema", center_idx=20,
                 direction=1, start_idx=21, end_idx=25, resolved=True),
            dict(roi_id="ROI_0", detector_name="derivative_extrema", center_idx=24,
                 direction=-1, start_idx=19, end_idx=21, resolved=True)]
    _split_extrema_overlaps(rows, signal, fs=100,
                            noise={"sigma_udeco": .01, "sigma_du": .001},
                            config=ExperimentConfig())
    assert rows[0]["end_idx"] == rows[1]["start_idx"] == 21
    assert rows[0]["start_idx"] == 17
    assert rows[1]["end_idx"] == 26
    assert rows[0]["start_rule"] == "overlap_third_reversal"
    assert rows[1]["end_rule"] == "overlap_third_reversal"
    assert all(row["resolved"] for row in rows)


def test_fallback_boundary_stops_at_third_reverse_when_turn_does_not():
    from udeco_uh_roi_edge_detection import _fallback_boundary

    steps = np.zeros(20)
    steps[10:16] = [-.1, .05, -.1, .05, -.1, .05]
    values = np.r_[0, np.cumsum(steps)]
    assert _fallback_boundary(values, 10, 17, 1) == (15, "third_reversal")


def test_fallback_boundary_growing_reverse_still_precedes_third_reverse():
    from udeco_uh_roi_edge_detection import _fallback_boundary

    steps = np.zeros(20)
    steps[10:13] = [-.1, .2, -.1]
    values = np.r_[0, np.cumsum(steps)]
    assert _fallback_boundary(values, 10, 14, 1) == (11, "turn")


def test_roi_mode_uses_rising_shape_and_keeps_falling_shape_diagnostic():
    rois = [{"roi_id": "ROI_0", "left_idx": 100, "right_idx": 200},
            {"roi_id": "ROI_1", "left_idx": 201, "right_idx": 300}]
    table = _direct_table([
        _direct_edge("ROI_0", 120, 1, delta=.01, is_sharp=True),
        _direct_edge("ROI_0", 150, -1, delta=-.01),
        _direct_edge("ROI_1", 220, 1, delta=.01),
        _direct_edge("ROI_1", 250, -1, delta=-.01, is_sharp=True)], rois)
    assert table.roi_mode.tolist() == ["detach-attach", "detach-attach",
                                      "release-force", "release-force"]
    assert table.morphology.tolist() == ["sharp", "gradual", "gradual", "sharp"]
    assert table.event_label.tolist() == ["detach", "attach", "release", "force"]


def test_roi_mode_requires_rising_start_before_falling_start_including_fallback():
    rois = [{"roi_id": "ROI_0", "left_idx": 100, "right_idx": 200},
            {"roi_id": "ROI_1", "left_idx": 201, "right_idx": 300}]
    fall_first = _direct_edge("ROI_0", 120, -1, delta=-.01)
    rise_later = _direct_edge("ROI_0", 150, 1, delta=.01, is_sharp=True)
    fallback_rise = _direct_edge("ROI_1", 250, 1, delta=.01, is_sharp=True)
    fallback_fall = _direct_edge("ROI_1", 250, -1, delta=-.01)
    for row in (fallback_rise, fallback_fall):
        row["detector_name"] = "peak_fallback"
    fallback_rise["start_idx"] = 220
    fallback_fall["start_idx"] = 250
    table = _direct_table([fall_first, rise_later, fallback_rise, fallback_fall], rois)
    assert table.roi_mode.tolist() == ["release-force", "release-force",
                                      "detach-attach", "detach-attach"]
    assert table.event_label.tolist() == ["force", "release", "detach", "attach"]


def test_delta_uses_final_edge_range_and_endpoint_direction_without_plateaus():
    edge = np.zeros(2200)
    edge[1800:1807] = np.arange(1, 8) * .001
    edge[1807:1814] = .007 - np.arange(1, 8) * .001
    roi = {"roi_id": "ROI_0", "left_idx": 1795, "search_left_idx": 1792,
           "right_idx": 1815, "main_peak_idx": 1806}
    cfg = ExperimentConfig()
    candidates, _ = detect_roi_candidates(edge, [roi], fs=100, config=cfg)
    # The clean trace has a different scale; delta must use edge_signal.
    rows = localize_roi_candidates(edge * 10, edge, candidates, [roi], fs=100,
                                   noise={"sigma_udeco": 1e-4, "sigma_du": 1e-4}, config=cfg)
    assert len(rows) == 2
    assert all(row["resolved"] for row in rows)
    assert all(np.isclose(row["delta"], np.ptp(edge[row["start_idx"]:row["end_idx"] + 1]))
               for row in rows)
    assert all(row["direction"] == np.sign(edge[row["end_idx"]] - edge[row["start_idx"]])
               for row in rows)
    assert all("settle_end_idx" not in row for row in rows)


def test_edge_range_amplitude_can_exceed_endpoint_change_and_zero_change_fails():
    from udeco_uh_roi_edge_detection import _measure_resolved_edge

    signal = np.array([0.0, 2.0, 1.0])
    row = {"detector_name": "derivative_extrema", "direction": 1,
           "start_idx": 0, "end_idx": 2, "resolved": True}
    noise = {"sigma_udeco": .5, "sigma_du": .1}
    _measure_resolved_edge(row, signal, fs=100, noise=noise, config=ExperimentConfig())
    assert row["resolved"] and row["direction"] == 1
    assert row["delta"] == row["amplitude"] == 2.0
    assert row["endpoint_change"] == 1.0
    assert row["baseline_snr"] == 4.0

    flat_end = {**row, "end_idx": 2}
    _measure_resolved_edge(flat_end, np.array([0.0, 2.0, 0.0]), fs=100,
                           noise=noise, config=ExperimentConfig())
    assert not flat_end["resolved"] and flat_end["reason"] == "zero_endpoint_change"


def test_weak_peak_and_unfinished_edge_keep_failure_reason():
    signal = np.zeros(2200)
    signal[1800] = .0001
    roi = {"roi_id": "ROI_0", "left_idx": 1795, "right_idx": 1810}
    candidate = {"roi_id": "ROI_0", "center_idx": 1800, "direction": 1,
                 "detector_name": "derivative_extrema"}
    noise = {"sigma_udeco": 1e-4, "sigma_du": 1e-4}
    weak = localize_roi_candidates(signal, signal, [candidate], [roi], fs=100,
                                   noise=noise, config=ExperimentConfig())[0]
    assert weak["reason"] == "weak_derivative"
    rising = np.zeros(2200)
    rising[1800:] = np.arange(400) * .001
    unfinished = localize_roi_candidates(rising, rising, [{**candidate, "center_idx": 1801}], [roi], fs=100,
                                         noise=noise, config=ExperimentConfig(extrema_trace_sec=.1))[0]
    assert unfinished["reason"] == "core_does_not_end"


def test_peak_fallback_positive_and_negative_turn_beyond_roi():
    from udeco_uh_roi_edge_detection import recover_missing_extrema_pairs
    positive = np.zeros(100)
    positive[18] = .02
    positive[20:30] = np.arange(1, 11) * .01
    positive[30:40] = np.arange(9, -1, -1) * .01
    positive[40] = .02
    roi = {"roi_id": "ROI_0", "left_idx": 23, "search_left_idx": 0,
           "right_idx": 35}
    noise = {"sigma_udeco": 1e-4, "sigma_du": 1e-4}
    for signal, directions in ((positive, (1, -1)), (-positive, (-1, 1))):
        rows = recover_missing_extrema_pairs(signal, [], [roi], fs=100,
                                              noise=noise, config=ExperimentConfig())
        assert len(rows) == 2
        assert [row["direction"] for row in rows] == list(directions)
        assert [(row["start_idx"], row["end_idx"]) for row in rows] == [(19, 29), (29, 39)]
        assert all(row["endpoint_source"] == "turn" for row in rows)
        assert rows[0]["start_idx"] < roi["left_idx"]
        assert rows[1]["end_idx"] > roi["right_idx"]


def test_fallback_search_limit_and_main_pair_switch():
    from udeco_uh_roi_edge_detection import recover_missing_extrema_pairs
    signal = np.zeros(100)
    signal[20:31] = np.arange(1, 12) * .01
    signal[31:] = signal[30]
    roi = {"roi_id": "ROI_0", "left_idx": 24, "search_left_idx": 0,
           "right_idx": 33}
    noise = {"sigma_udeco": 1e-4, "sigma_du": 1e-4}
    rows = recover_missing_extrema_pairs(signal, [], [roi], fs=100,
                                          noise=noise, config=ExperimentConfig(extrema_trace_sec=.1))
    assert rows[0]["search_limit"] and rows[0]["resolved"]
    assert rows[1]["search_limit"] and not rows[1]["resolved"]
    original = [_direct_edge("ROI_0", 25, 1, delta=.01),
                _direct_edge("ROI_0", 32, -1, resolved=False)]
    positive = np.zeros(100)
    positive[18] = .02
    positive[20:30] = np.arange(1, 11) * .01
    positive[30:40] = np.arange(9, -1, -1) * .01
    positive[40] = .02
    recovered = recover_missing_extrema_pairs(positive, original, [roi], fs=100,
                                               noise=noise, config=ExperimentConfig())
    table = _direct_table(recovered, [roi])
    assert table.loc[table.detector_name == "peak_fallback", "is_main_rising"].any()
    assert table.loc[table.detector_name == "peak_fallback", "is_main_falling"].any()
    assert not table.loc[table.detector_name == "derivative_extrema", "is_main_rising"].any()
    assert set(table.pair_status) == {"fallback_localized"}
    complete = [_direct_edge("ROI_0", 25, 1, delta=.01),
                _direct_edge("ROI_0", 32, -1, delta=-.01)]
    assert len(recover_missing_extrema_pairs(positive, complete, [roi], fs=100,
                                             noise=noise, config=ExperimentConfig())) == 2


def _extrema_from_steps(steps, peak_k=10, config=None, direction=1):
    signal = np.r_[0.0, np.cumsum(steps)]
    roi = {"roi_id": "ROI_0", "left_idx": 1, "right_idx": len(signal)-2}
    candidate = {"roi_id": "ROI_0", "center_idx": peak_k+1, "direction": direction,
                 "detector_name": "derivative_extrema"}
    return localize_roi_candidates(signal, signal, [candidate], [roi], fs=100,
                                   noise={"sigma_udeco": .01, "sigma_du": .01},
                                   config=config or ExperimentConfig())[0]


def test_sharp_positive_uses_either_neighbor_and_distinct_side_thresholds():
    steps = np.zeros(30)
    steps[8:14] = [.01, .02, .2, .12, .02, .01]
    row = _extrema_from_steps(steps, config=ExperimentConfig(sharp_peak_sigma_k=10))
    assert row["resolved"]
    assert row["right_mode"] == "positive_sharp"
    assert (row["start_idx"], row["end_idx"]) == (10, 12)
    assert (row["start_rule"], row["end_rule"]) == ("threshold", "threshold")
    assert np.isclose(row["threshold_left"], .03)
    assert np.isclose(row["threshold_right"], .08)
    assert row["is_sharp"]


def test_general_edge_uses_three_low_left_and_four_low_right():
    steps = np.zeros(30)
    steps[6:16] = [.01, .01, .01, .12, .2, .12, .01, .01, .01, .01]
    positive = _extrema_from_steps(steps)
    negative = _extrema_from_steps(-steps, direction=-1)
    for row in (positive, negative):
        assert row["resolved"]
        assert (row["start_idx"], row["end_idx"]) == (8, 15)
        assert (row["threshold_left"], row["threshold_right"]) == (.02, .02)
        assert not row["is_sharp"]


def test_general_edge_stops_before_third_nonconsecutive_reverse_sample():
    steps = np.zeros(30)
    steps[5:17] = [-.04, .12, -.04, .12, -.04, .2,
                   .12, -.04, .12, -.04, .12, -.04]
    row = _extrema_from_steps(steps)
    assert row["resolved"]
    assert (row["start_idx"], row["end_idx"]) == (6, 16)
    assert (row["start_rule"], row["end_rule"]) == ("third_reversal", "third_reversal")


def test_sharp_left_zero_difference_stops_at_peak_boundary():
    steps = np.zeros(30)
    steps[9:14] = [0, .2, .12, .01, .01]
    row = _extrema_from_steps(steps, config=ExperimentConfig(sharp_peak_sigma_k=10))
    assert row["resolved"]
    assert row["start_idx"] == 10
    assert row["start_rule"] == "nonpositive_adjacent"


def test_sharp_four_neighbors_and_strict_baseline_peak_floor():
    steps = np.zeros(30)
    steps[8:13] = [.01, .2, .4, .01, .01]
    one_strong = _extrema_from_steps(steps)
    assert one_strong["is_sharp"]
    assert one_strong["strong_neighbor_count"] == 1
    assert one_strong["right_mode"] == "positive_sharp"
    assert _extrema_from_steps(-steps, direction=-1)["is_sharp"]

    steps[8] = .2
    two_strong = _extrema_from_steps(steps)
    assert not two_strong["is_sharp"]
    assert two_strong["strong_neighbor_count"] == 2

    steps[8:13] = [.01, .01, .3, .01, .01]
    assert not _extrema_from_steps(steps)["is_sharp"]
    steps[10] = .31
    assert _extrema_from_steps(steps)["is_sharp"]


def test_sharp_nonpositive_start_uses_plot_aligned_adjacent_or_next_point():
    adjacent = np.zeros(30)
    adjacent[9:13] = [-.01, .4, .01, .01]
    row = _extrema_from_steps(adjacent)
    assert (row["start_idx"], row["start_rule"]) == (10, "nonpositive_adjacent")

    nonadjacent = adjacent.copy()
    nonadjacent[8:10] = [-.01, .005]
    row = _extrema_from_steps(nonadjacent)
    assert (row["start_idx"], row["start_rule"]) == (10, "nonpositive_nonadjacent")


def test_new_threshold_parameters_override_each_mode_independently():
    sharp_steps = np.zeros(30)
    sharp_steps[8:14] = [.01, .02, .2, .12, .02, .01]
    config = ExperimentConfig(sharp_peak_sigma_k=10, threshold_du_sharp_left=4,
                              threshold_dupeak_sharp_left=.25,
                              threshold_du_sharp_right=5,
                              threshold_dupeak_sharp_right=.5)
    sharp = _extrema_from_steps(sharp_steps, config=config)
    assert np.isclose(sharp["threshold_left"], .05)
    assert np.isclose(sharp["threshold_right"], .1)

    general_steps = np.zeros(30)
    general_steps[6:16] = [.01, .01, .01, .12, .2, .12, .01, .01, .01, .01]
    config = ExperimentConfig(threshold_du_general_left=4,
                              threshold_dupeak_general_left=.1,
                              threshold_du_general_right=3,
                              threshold_dupeak_general_right=.2)
    general = _extrema_from_steps(general_steps, config=config)
    assert np.isclose(general["threshold_left"], .04)
    assert np.isclose(general["threshold_right"], .04)


def test_peak_fallback_selects_mean_centered_extreme():
    from udeco_uh_roi_edge_detection import recover_missing_extrema_pairs
    values = np.array([1.0, 1.0, 1.01, 1.04, 1.1, 1.04, 1.01, 1.0, 1.0])
    roi = {"roi_id": "ROI_0", "left_idx": 0, "right_idx": 8}
    noise = {"sigma_udeco": .001, "sigma_du": .001}
    for offset in (0, 5):
        rows = recover_missing_extrema_pairs(values + offset, [], [roi], fs=100,
                                              noise=noise, config=ExperimentConfig())
        assert {row["fallback_peak_idx"] for row in rows} == {4}


def test_roi_classification_requires_amplitude_matched_qualified_pair():
    rois = [{"roi_id": "ROI_0", "left_idx": 100, "right_idx": 200}]
    for rise_delta, fall_delta, status in (
            (.01, -.005, "classified"),
            (.01, -.0049, "roi_delta_mismatch"),
            (.01, -.0002, "roi_missing_pair")):
        table = _direct_table([_direct_edge("ROI_0", 130, 1, delta=rise_delta),
                               _direct_edge("ROI_0", 170, -1, delta=fall_delta)], rois)
        assert set(table.classification_status) == {status}
        assert table.morphology.notna().all() == (status == "classified")
        assert table.event_label.notna().all() == (status == "classified")
        assert table.is_main_rising.sum() == table.is_main_falling.sum() == 1
    single = _direct_table([_direct_edge("ROI_0", 130, 1, delta=.01)], rois)
    assert single.classification_status.item() == "roi_missing_pair"
    assert single.event_label.isna().all()


def test_fallback_main_pair_controls_roi_classification():
    rois = [{"roi_id": "ROI_0", "left_idx": 100, "right_idx": 200}]
    original = [_direct_edge("ROI_0", 125, 1, delta=.02),
                _direct_edge("ROI_0", 175, -1, resolved=False)]
    fallback = [_direct_edge("ROI_0", 140, 1, delta=.01),
                _direct_edge("ROI_0", 160, -1, delta=-.006)]
    for row in fallback:
        row["detector_name"] = "peak_fallback"
    table = _direct_table(original + fallback, rois)
    assert set(table.main_source) == {"peak_fallback"}
    assert table.loc[table.detector_name == "peak_fallback", "event_label"].notna().all()
    original_rise = table.loc[(table.detector_name == "derivative_extrema")
                              & (table.direction == 1)].iloc[0]
    assert original_rise.classification_status == "not_main"
    assert original_rise.event_label is None
    assert original_rise.morphology is None
    fallback[1]["delta"] = .004
    mismatched = _direct_table(original + fallback, rois)
    assert set(mismatched.classification_status) == {"roi_delta_mismatch"}
    assert mismatched.event_label.isna().all()
