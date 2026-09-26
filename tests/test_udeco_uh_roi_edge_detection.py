"""Core regression tests for the Uh-guided Udeco experiment."""

from pathlib import Path

import numpy as np

from udeco_uh_roi_edge_detection import (
    ExperimentConfig,
    baseline_sigmas,
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
