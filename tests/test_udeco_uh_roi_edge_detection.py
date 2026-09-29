"""Core regression tests for the Uh-guided Udeco experiment."""

from pathlib import Path

import numpy as np

from udeco_uh_roi_edge_detection import (
    ExperimentConfig,
    baseline_sigmas,
    build_direct_event_table,
    build_event_table,
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


def test_median_difference_detects_one_sample_steps_only_inside_roi():
    rng = np.random.default_rng(12)
    signal = rng.normal(0, 1e-4, 2500)
    signal[1800:1900] += 0.01
    signal[2200:] += 0.02  # Must not be detected outside the Uh ROI.
    roi = {"roi_id": "ROI_0", "left_idx": 1750, "right_idx": 1950}
    candidates, diagnostics = detect_roi_candidates(
        signal, [roi], fs=100, config=ExperimentConfig())
    assert any(c["direction"] == 1 and abs(c["center_idx"] - 1800) <= 1 for c in candidates)
    assert any(c["direction"] == -1 and abs(c["center_idx"] - 1900) <= 1 for c in candidates)
    assert all(1750 <= c["center_idx"] <= 1950 for c in candidates)
    assert all(c["detector_name"] == "median_diff" for c in candidates)
    assert set(("roi_id", "center_idx", "region_start_idx", "region_end_idx",
                "direction", "detector_score", "detector_name")) <= candidates[0].keys()
    assert 1 in diagnostics["scale_scores"]


def test_udeco_detection_extends_only_the_left_side_of_logical_roi():
    signal = np.zeros(2500)
    signal[1798:1820] = 0.01
    roi = {"roi_id": "ROI_0", "left_idx": 1800, "right_idx": 1850,
           "search_left_idx": 1797}
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
           "right_idx": 1850}

    candidates, diagnostics = detect_roi_candidates(
        signal, [roi], fs=100,
        config=ExperimentConfig(edge_detector="derivative_extrema"))

    assert [(row["center_idx"], row["direction"]) for row in candidates] == [
        (1798, 1), (1820, -1)]
    assert all(row["region_start_idx"] == row["center_idx"] == row["region_end_idx"]
               for row in candidates)
    assert all(row["detector_name"] == "derivative_extrema" for row in candidates)
    sigma_du = np.std(np.diff(signal[:1500]))
    assert np.isclose(diagnostics["derivative_sigma"], sigma_du)
    assert np.isclose(candidates[0]["detector_score"], 0.01 / sigma_du)


def test_derivative_extrema_keeps_only_available_direction_without_score_gate():
    rng = np.random.default_rng(24)
    signal = np.zeros(2500)
    signal[:1500] = rng.normal(0, 1e-4, 1500)
    signal[1800:1860] = np.arange(60) * 1e-5
    roi = {"roi_id": "ROI_0", "left_idx": 1801, "right_idx": 1850}

    candidates, _ = detect_roi_candidates(
        signal, [roi], fs=100,
        config=ExperimentConfig(edge_detector="derivative_extrema"))

    assert len(candidates) == 1
    assert candidates[0]["direction"] == 1
    assert candidates[0]["detector_score"] < 3.0


def test_haar_switch_uses_same_candidate_contract_and_fixed_baseline_noise():
    rng = np.random.default_rng(13)
    signal = rng.normal(0, 1e-4, 2500)
    signal[1800:1900] += 0.01
    roi = {"roi_id": "ROI_0", "left_idx": 1750, "right_idx": 1950}
    cfg = ExperimentConfig(edge_detector="haar")
    candidates, diagnostics = detect_roi_candidates(signal, [roi], fs=100, config=cfg)
    assert any(c["direction"] == 1 and abs(c["center_idx"] - 1800) <= 5 for c in candidates)
    assert any(c["direction"] == -1 and abs(c["center_idx"] - 1900) <= 5 for c in candidates)
    assert all(c["detector_name"] == "haar" for c in candidates)
    assert all(c["roi_id"] == "ROI_0" for c in candidates)
    assert diagnostics["scale_sigmas"]
    altered = signal.copy()
    altered[2050:] += rng.normal(0, 0.1, len(altered[2050:]))
    _, altered_diagnostics = detect_roi_candidates(altered, [roi], fs=100, config=cfg)
    assert diagnostics["scale_sigmas"] == altered_diagnostics["scale_sigmas"]


def _localized_synthetic(signal, center, *, fs=100):
    roi = {"roi_id": "ROI_0", "left_idx": center - 45, "right_idx": center + 80}
    candidate = {"roi_id": "ROI_0", "center_idx": center, "direction": 1,
                 "region_start_idx": center, "region_end_idx": center,
                 "detector_score": 20.0, "detector_name": "median_diff"}
    noise = {"sigma_udeco": 1e-4, "sigma_du": 1e-4}
    return localize_roi_candidates(signal, signal, [candidate], [roi],
                                   fs=fs, noise=noise, config=ExperimentConfig())[0]


def test_localization_separates_one_cycle_step_from_seventy_ms_ramp():
    step = np.zeros(2200)
    step[1800:] = 0.01
    sharp = _localized_synthetic(step, 1800)
    assert sharp["resolved"]
    assert sharp["start_idx"] == 1799
    assert sharp["end_idx"] == 1800
    assert sharp["T_edge"] == 0.01
    assert sharp["derivative_peak_width"] == 0.01

    ramp = np.zeros(2200)
    ramp[1800:1807] = np.arange(1, 8) * 0.01 / 7
    ramp[1807:] = 0.01
    gradual = _localized_synthetic(ramp, 1803)
    assert gradual["resolved"]
    assert abs(gradual["start_idx"] - 1799) <= 1
    assert abs(gradual["T_edge"] - 0.07) <= 0.01
    assert gradual["derivative_peak_width"] >= 0.06


def test_localization_can_finish_beyond_logical_roi_right_boundary():
    signal = np.zeros(2200)
    signal[1848:1858] = np.arange(1, 11) * 0.001
    signal[1858:] = 0.01
    roi = {"roi_id": "ROI_0", "left_idx": 1800,
           "search_left_idx": 1797, "right_idx": 1850}
    candidate = {"roi_id": "ROI_0", "center_idx": 1849, "direction": 1,
                 "region_start_idx": 1849, "region_end_idx": 1849,
                 "detector_score": 20.0, "detector_name": "median_diff"}
    measured = localize_roi_candidates(
        signal, signal, [candidate], [roi], fs=100,
        noise={"sigma_udeco": 1e-4, "sigma_du": 1e-4},
        config=ExperimentConfig())[0]
    assert measured["resolved"]
    assert measured["end_idx"] > roi["right_idx"]


def test_extrema_edge_can_end_far_beyond_roi_search_boundary():
    signal = np.zeros(2200)
    signal[1849] = 0.002
    signal[1850:1906] = 0.002 + np.arange(1, 57) * 0.0005
    signal[1906:] = signal[1905]
    roi = {"roi_id": "ROI_0", "left_idx": 1800,
           "search_left_idx": 1797, "right_idx": 1850}
    candidates, _ = detect_roi_candidates(
        signal, [roi], fs=100,
        config=ExperimentConfig(edge_detector="derivative_extrema"))
    rising = next(row for row in candidates if row["direction"] == 1)
    measured = localize_roi_candidates(
        signal, signal, [rising], [roi], fs=100,
        noise={"sigma_udeco": 1e-4, "sigma_du": 1e-4},
        config=ExperimentConfig())[0]
    assert rising["center_idx"] == 1849
    assert measured["resolved"]
    assert measured["start_idx"] == 1848
    assert measured["end_idx"] > roi["right_idx"] + 20
    assert np.isclose(measured["core_threshold"], 0.0003)


def test_localization_can_start_before_left_search_boundary():
    signal = np.zeros(2200)
    signal[1794:1802] = np.arange(1, 9) * 0.001
    signal[1802:] = 0.008
    roi = {"roi_id": "ROI_0", "left_idx": 1800,
           "search_left_idx": 1797, "right_idx": 1850}
    candidate = {"roi_id": "ROI_0", "center_idx": 1798, "direction": 1,
                 "region_start_idx": 1798, "region_end_idx": 1798,
                 "detector_score": 20.0, "detector_name": "median_diff"}
    measured = localize_roi_candidates(
        signal, signal, [candidate], [roi], fs=100,
        noise={"sigma_udeco": 1e-4, "sigma_du": 1e-4},
        config=ExperimentConfig())[0]
    assert measured["resolved"]
    assert measured["start_idx"] < roi["search_left_idx"]


def test_fast_step_with_long_tail_is_classified_by_edge_duration():
    signal = np.zeros(2200)
    signal[1800:] = 0.01
    signal[1801:1822] += np.arange(1, 22) * 0.0003
    signal[1822:] += 21 * 0.0003
    localized = _localized_synthetic(signal, 1800)
    table = build_event_table(
        [localized], [{"roi_id": "ROI_0", "left_idx": 1755,
                       "main_peak_idx": 1810, "right_idx": 1880}], channel="Udeco2",
        sigma_udeco=1e-4, fs=100, config=ExperimentConfig(),
        direction_map={1: {"sharp": "detach", "gradual": "release"},
                       -1: {"sharp": "attach", "gradual": "pressure"}})
    assert localized["T_edge"] == 0.01
    assert "T_tail" not in localized
    assert table.iloc[0]["morphology"] == "sharp"
    assert table.iloc[0]["event_label"] is None


def test_roi_requires_left_then_peak_opposite_edges_and_prefers_peak_amplitude():
    base = {"roi_id": "ROI_0", "resolved": True, "T_edge": 0.01,
            "direction": 1, "derivative_peak": 0.002,
            "normalized_derivative_peak": 20.0, "derivative_peak_width": 0.01,
            "monotonicity": 1.0, "detector_name": "median_diff"}
    localized = [{**base, "center_idx": 105, "start_idx": 105, "delta": 0.01},
                 {**base, "center_idx": 145, "start_idx": 145, "delta": 0.02},
                 {**base, "center_idx": 165, "start_idx": 165,
                  "direction": -1, "delta": -0.008},
                 {**base, "center_idx": 205, "start_idx": 205,
                  "direction": -1, "delta": -0.03}]
    table = build_event_table(
        localized, [{"roi_id": "ROI_0", "left_idx": 100, "search_left_idx": 97,
                     "main_peak_idx": 160, "right_idx": 220}], channel="Udeco2",
        sigma_udeco=1e-4, fs=100, config=ExperimentConfig(),
        direction_map={1: {"sharp": "detach", "gradual": "release"},
                       -1: {"sharp": "attach", "gradual": "pressure"}})
    assert table.loc[table.is_main_rising, "center_idx"].tolist() == [145]
    assert table.loc[table.is_main_falling, "center_idx"].tolist() == [205]
    assert table.loc[table.center_idx == 165, "is_main_falling"].item() == False
    assert set(table.loc[table.edge_role.notna(), "edge_role"]) == {"left", "peak"}


def test_cross_roi_duplicate_prefers_later_roi_and_missing_pair_stays_unpaired():
    base = {"resolved": True, "T_edge": 0.01,
            "derivative_peak_width": 0.01, "normalized_derivative_peak": 20.0,
            "detector_name": "median_diff"}
    localized = [{**base, "roi_id": "ROI_0", "center_idx": 199,
                  "start_idx": 199, "direction": 1, "delta": 0.01},
                 {**base, "roi_id": "ROI_1", "center_idx": 199,
                  "start_idx": 199, "direction": 1, "delta": 0.01},
                 {**base, "roi_id": "ROI_1", "center_idx": 249,
                  "start_idx": 249, "direction": -1, "delta": -0.01}]
    rois = [{"roi_id": "ROI_0", "left_idx": 100, "right_idx": 200,
             "main_peak_idx": 150},
            {"roi_id": "ROI_1", "left_idx": 201, "search_left_idx": 198,
             "right_idx": 300, "main_peak_idx": 250}]
    table = build_event_table(
        localized, rois, channel="Udeco2", sigma_udeco=1e-4,
        fs=100, config=ExperimentConfig(),
        direction_map={1: {"sharp": "detach", "gradual": "release"},
                       -1: {"sharp": "attach", "gradual": "pressure"}})
    assert table.loc[table.roi_id == "ROI_0", "overlap_duplicate"].item() == True
    assert table.loc[table.roi_id == "ROI_0", "is_main_rising"].item() == False
    assert table.loc[table.is_main_rising, "roi_id"].tolist() == ["ROI_1"]
    assert table.loc[table.is_main_falling, "roi_id"].tolist() == ["ROI_1"]


def _direct_edge(roi_id, center, direction, *, delta=None, resolved=True):
    row = {"roi_id": roi_id, "center_idx": center, "direction": direction,
           "region_start_idx": center, "region_end_idx": center,
           "detector_name": "derivative_extrema", "detector_score": 8.0,
           "resolved": resolved, "reason": None if resolved else "core_does_not_end"}
    if resolved:
        row.update(start_idx=center - 1, end_idx=center,
                   delta=delta, T_edge=0.01,
                   derivative_peak_width=0.01, normalized_derivative_peak=8.0)
    return row


def _direct_table(localized, rois):
    return build_direct_event_table(
        localized, rois, channel="Udeco2", sigma_udeco=1e-4, fs=100,
        config=ExperimentConfig(),
        direction_map={1: {"sharp": "detach", "gradual": "release"},
                       -1: {"sharp": "attach", "gradual": "pressure"}})


def test_direct_selection_displays_both_extrema_without_position_pair_gate():
    rois = [{"roi_id": "ROI_0", "left_idx": 100, "search_left_idx": 97,
             "main_peak_idx": 160, "right_idx": 220}]
    table = _direct_table([
        _direct_edge("ROI_0", 105, 1, delta=0.01),
        _direct_edge("ROI_0", 180, -1, resolved=False)], rois)
    assert table.is_main_rising.tolist() == [True, False]
    assert table.is_main_falling.tolist() == [False, True]
    assert table.pair_status.tolist() == ["unpaired", "unpaired"]
    assert table.event_label.tolist() == ["detach", None]
    assert table.classification_status.tolist() == ["classified", "localization_failed"]


def test_direct_classification_uses_strict_absolute_delta_and_flags_sign_conflict():
    rois = [{"roi_id": "ROI_0", "left_idx": 100, "right_idx": 200},
            {"roi_id": "ROI_1", "left_idx": 201, "right_idx": 300}]
    table = _direct_table([
        _direct_edge("ROI_0", 150, 1, delta=0.0002),
        _direct_edge("ROI_1", 250, 1, delta=-0.0003)], rois)
    assert table.pair_status.tolist() == ["unpaired", "unpaired"]
    assert table.event_label.tolist() == [None, "detach"]
    assert table.classification_status.tolist() == ["low_amplitude", "classified"]
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
    assert table.event_label.tolist() == ["detach", "detach"]


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


def test_nearby_candidate_centers_localize_distinct_steps():
    signal = np.zeros(2200)
    signal[1800:] += 0.01
    signal[1810:] += 0.02
    roi = {"roi_id": "ROI_0", "left_idx": 1750, "right_idx": 1880}
    candidates = [{"roi_id": "ROI_0", "center_idx": center, "direction": 1,
                   "region_start_idx": center, "region_end_idx": center,
                   "detector_score": 20.0, "detector_name": "median_diff"}
                  for center in (1800, 1810)]
    measured = localize_roi_candidates(
        signal, signal, candidates, [roi], fs=100,
        noise={"sigma_udeco": 1e-4, "sigma_du": 1e-4}, config=ExperimentConfig())
    assert measured[0]["derivative_peak_idx"] == 1800
    assert measured[0]["resolved"]
    assert np.isclose(measured[0]["delta"], 0.01)
    assert measured[1]["resolved"]
    assert measured[1]["derivative_peak_idx"] == 1810
    assert np.isclose(measured[1]["delta"], 0.02)


def test_extrema_localization_keeps_roi_anchor_when_stronger_peak_is_outside():
    signal = np.zeros(2200)
    signal[1800:] = .01
    signal[1810:] += .02
    roi = {"roi_id": "ROI_0", "left_idx": 1798, "search_left_idx": 1797, "right_idx": 1801}
    cfg = ExperimentConfig(edge_detector="derivative_extrema")
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
    assert first["end_idx"] < second["start_idx"]
    assert first["T_edge"] == .02
    assert first["delta"] == 1
    assert np.isclose(second["delta"], -.8)


def test_delta_uses_edge_signal_endpoints_without_stable_plateaus():
    edge = np.zeros(2200)
    edge[1800:1807] = np.arange(1, 8) * .001
    edge[1807:1814] = .007 - np.arange(1, 8) * .001
    roi = {"roi_id": "ROI_0", "left_idx": 1795, "search_left_idx": 1792,
           "right_idx": 1815}
    cfg = ExperimentConfig(edge_detector="derivative_extrema")
    candidates, _ = detect_roi_candidates(edge, [roi], fs=100, config=cfg)
    # The clean trace has a different scale; delta must use edge_signal.
    rows = localize_roi_candidates(edge * 10, edge, candidates, [roi], fs=100,
                                   noise={"sigma_udeco": 1e-4, "sigma_du": 1e-4}, config=cfg)
    assert len(rows) == 2
    assert all(row["resolved"] for row in rows)
    assert all(np.isclose(row["delta"], edge[row["end_idx"]] - edge[row["start_idx"]])
               for row in rows)
    assert all("settle_end_idx" not in row for row in rows)


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


def test_left_start_requires_two_quiet_differences_and_threshold_override():
    signal = np.zeros(2200)
    signal[1798:] += .0001
    signal[1799:] += .001
    signal[1800:] += .01
    roi = {"roi_id": "ROI_0", "left_idx": 1790, "right_idx": 1820}
    candidate = {"roi_id": "ROI_0", "center_idx": 1800, "direction": 1,
                 "detector_name": "derivative_extrema"}
    cfg = ExperimentConfig(threshold_du=2, threshold_dupeak=.15)
    row = localize_roi_candidates(signal, signal, [candidate], [roi], fs=100,
                                  noise={"sigma_udeco": 1e-4, "sigma_du": 1e-4}, config=cfg)[0]
    assert row["resolved"]
    assert row["start_idx"] == 1799
    assert np.isclose(row["core_threshold"], .0015)
    stronger_floor = ExperimentConfig(threshold_du=200, threshold_dupeak=.15)
    weak = localize_roi_candidates(signal, signal, [candidate], [roi], fs=100,
                                   noise={"sigma_udeco": 1e-4, "sigma_du": 1e-4},
                                   config=stronger_floor)[0]
    assert weak["reason"] == "weak_derivative"


def test_peak_fallback_positive_and_negative_turn_beyond_roi():
    from udeco_uh_roi_edge_detection import recover_missing_extrema_pairs
    positive = np.zeros(100)
    positive[18] = .02
    positive[20:30] = np.arange(1, 11) * .01
    positive[30:40] = np.arange(9, -1, -1) * .01
    positive[40] = .02
    roi = {"roi_id": "ROI_0", "left_idx": 23, "search_left_idx": 22,
           "right_idx": 35}
    noise = {"sigma_udeco": 1e-4, "sigma_du": 1e-4}
    for signal, directions in ((positive, (1, -1)), (-positive, (-1, 1))):
        rows = recover_missing_extrema_pairs(signal, [], [roi], fs=100,
                                              noise=noise, config=ExperimentConfig())
        assert len(rows) == 2
        assert [row["direction"] for row in rows] == list(directions)
        assert [(row["start_idx"], row["end_idx"]) for row in rows] == [(19, 29), (29, 39)]
        assert all(row["endpoint_source"] == "turn" for row in rows)
        assert rows[0]["start_idx"] < roi["search_left_idx"]
        assert rows[1]["end_idx"] > roi["right_idx"]


def test_fallback_search_limit_and_main_pair_switch():
    from udeco_uh_roi_edge_detection import recover_missing_extrema_pairs
    signal = np.zeros(100)
    signal[20:31] = np.arange(1, 12) * .01
    signal[31:] = signal[30]
    roi = {"roi_id": "ROI_0", "left_idx": 24, "search_left_idx": 22,
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
