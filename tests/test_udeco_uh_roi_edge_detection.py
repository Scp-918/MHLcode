"""Core regression tests for the Uh-guided Udeco experiment."""

from pathlib import Path

import numpy as np

from udeco_uh_roi_edge_detection import (
    ExperimentConfig,
    baseline_sigmas,
    build_event_table,
    detect_roi_candidates,
    find_uh_rois,
    load_paired_sample,
    localize_roi_candidates,
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


def test_overlapping_peak_rois_merge_and_keep_source_peaks():
    x = np.zeros(300)
    x[100:111] = np.linspace(0, 5, 11)
    x[111:122] = np.linspace(4.5, 0, 11)
    x[130:141] = np.linspace(0, 4, 11)
    x[141:152] = np.linspace(3.6, 0, 11)
    cfg = ExperimentConfig(
        uh_peak_prominence_sigma=2,
        uh_peak_distance_sec=0.05,
        uh_roi_margin_sec=0.20,
    )
    rois, peaks = find_uh_rois(x, fs=100, sigma_uh=0.1, sigma_duh=0.1, config=cfg)
    assert len(peaks) == 2
    assert len(rois) == 1
    assert rois[0]["roi_id"] == "ROI_0"
    assert len(rois[0]["peak_indices"]) == 2
    assert rois[0]["left_idx"] <= 100
    assert rois[0]["right_idx"] >= 151


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
    assert sharp["core_end_idx"] == 1800
    assert sharp["T_core"] == 0.01
    assert sharp["derivative_peak_width"] == 0.01

    ramp = np.zeros(2200)
    ramp[1800:1807] = np.arange(1, 8) * 0.01 / 7
    ramp[1807:] = 0.01
    gradual = _localized_synthetic(ramp, 1803)
    assert gradual["resolved"]
    assert abs(gradual["start_idx"] - 1799) <= 1
    assert abs(gradual["T_core"] - 0.07) <= 0.01
    assert gradual["derivative_peak_width"] >= 0.06


def test_fast_step_with_long_tail_stays_sharp():
    signal = np.zeros(2200)
    signal[1800:] = 0.01
    signal[1801:1822] += np.arange(1, 22) * 0.0003
    signal[1822:] += 21 * 0.0003
    localized = _localized_synthetic(signal, 1800)
    table = build_event_table(
        [localized], [{"roi_id": "ROI_0"}], channel="Udeco2",
        sigma_udeco=1e-4, config=ExperimentConfig(),
        direction_map={1: {"sharp": "detach", "gradual": "release"},
                       -1: {"sharp": "attach", "gradual": "pressure"}})
    assert localized["T_core"] == 0.01
    assert localized["T_tail"] >= 0.10
    assert table.iloc[0]["morphology"] == "sharp_long_tail"
    assert table.iloc[0]["event_label"] == "detach"


def test_roi_selects_only_largest_actual_rise_and_fall():
    base = {"roi_id": "ROI_0", "resolved": True, "T_core": 0.01,
            "T_tail": 0.0, "direction": 1, "derivative_peak": 0.002,
            "normalized_derivative_peak": 20.0, "derivative_peak_width": 0.01,
            "monotonicity": 1.0, "detector_name": "median_diff"}
    localized = [{**base, "center_idx": 100, "delta": 0.001},
                 {**base, "center_idx": 110, "delta": 0.01},
                 {**base, "center_idx": 120, "direction": -1, "delta": -0.008}]
    table = build_event_table(
        localized, [{"roi_id": "ROI_0"}], channel="Udeco2",
        sigma_udeco=1e-4, config=ExperimentConfig(),
        direction_map={1: {"sharp": "detach", "gradual": "release"},
                       -1: {"sharp": "attach", "gradual": "pressure"}})
    assert table.loc[table.is_main_rising, "center_idx"].tolist() == [110]
    assert table.loc[table.is_main_falling, "center_idx"].tolist() == [120]
    assert table.loc[table.center_idx == 100, "is_main_rising"].item() == False


def test_slow_uh_peak_roi_contains_both_true_event_boundaries():
    uh = np.zeros(300)
    uh[100:151] = np.linspace(0, 5, 51)
    uh[151:202] = np.linspace(4.9, 0, 51)
    rois, _ = find_uh_rois(uh, fs=100, sigma_uh=0.1, sigma_duh=0.1,
                           config=ExperimentConfig())
    assert len(rois) == 1
    assert rois[0]["left_idx"] <= 100
    assert rois[0]["right_idx"] >= 201


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
