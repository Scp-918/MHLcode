"""Core regression tests for the Uh-guided Udeco experiment."""

from pathlib import Path

import numpy as np

from udeco_uh_roi_edge_detection import (
    ExperimentConfig,
    baseline_sigmas,
    detect_roi_candidates,
    find_uh_rois,
    load_paired_sample,
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
