from __future__ import annotations


from pathlib import Path


import numpy as np


import pandas as pd


import pytest


from scipy.io import savemat


from udeco_analysis import AnalysisConfig, PYDISPLAY_SENSOR_COLUMNS, detect_motion_stage, discover_subject_samples, load_calibration, load_udeco_sample, robust_sigma


def write_calibration(root: Path, *, c2: float = 2.0, c3: float = 3.0) -> Path:
    c_result = np.empty((3, 1), dtype=object)
    c_result[:, 0] = [
        {"channel": 1, "c": 1.0},
        {"channel": 2, "c": c2},
        {"channel": 3, "c": c3},
    ]
    k_result = np.empty((2, 1), dtype=object)
    k_result[:, 0] = [
        {"mainChannel": 2, "k": 0.1},
        {"mainChannel": 3, "k": 0.2},
    ]
    path = root / "ck.mat"
    savemat(path, {"cResult": c_result, "kResult": k_result})
    return path


def write_sensor(
    path: Path,
    *,
    absolute_seq: list[int] | None = None,
    parser_valid: list[bool] | None = None,
    n: int = 4,
) -> None:
    absolute_seq = absolute_seq or list(range(n))
    parser_valid = parser_valid or [True] * len(absolute_seq)
    rows = len(absolute_seq)
    frame = pd.DataFrame({name: np.ones(rows) for name in PYDISPLAY_SENSOR_COLUMNS})
    frame["frame_seq"] = np.asarray(absolute_seq) % 65536
    frame["absolute_seq_u64"] = absolute_seq
    frame["segment_id"] = 0
    frame["sample_seq"] = np.arange(rows) + 1
    frame["parser_valid"] = parser_valid
    frame["Uh2"] = 4.0
    frame["Uh3"] = 6.0
    frame["Uc2"] = 0.5
    frame["Uc3"] = 1.0
    frame["ACC_X"] = 0.0
    frame["ACC_Y"] = 0.0
    frame["ACC_Z"] = 1.0
    frame.to_csv(path, index=False)


def write_hr(path: Path) -> None:
    pd.DataFrame({"elapsed_seconds": [0.0, 1.0], "hr_bpm": [70, 71]}).to_csv(path, index=False)


def test_load_calibration_and_udeco_formula_with_gap_repair(tmp_path: Path) -> None:
    subject_dir = tmp_path / "ts"
    subject_dir.mkdir()
    write_calibration(tmp_path)
    sensor = subject_dir / "ts_write_1_sensor.csv"
    write_sensor(sensor, absolute_seq=[0, 2, 3], parser_valid=[True, True, False])

    calibration = load_calibration(tmp_path)
    sample = load_udeco_sample(sensor, data_root=tmp_path, fs=100)

    assert calibration.c2 == pytest.approx(2.0)
    assert calibration.c3 == pytest.approx(3.0)
    np.testing.assert_allclose(sample.time_s, [0.0, 0.01, 0.02])
    np.testing.assert_allclose(sample.udeco2, (4.0 - 0.5) / (2.0 - 0.5))
    np.testing.assert_allclose(sample.udeco3, (6.0 - 1.0) / (3.0 - 1.0))
    assert int(sample.qc["InterpFlag"].sum()) == 1
    assert sample.metadata["removed_parser_invalid_rows"] == 1
    assert sample.metadata["inserted_missing_rows"] == 1


def test_near_zero_udeco_denominator_is_interpolated_and_flagged(tmp_path: Path) -> None:
    subject_dir = tmp_path / "ts"
    subject_dir.mkdir()
    write_calibration(tmp_path)
    sensor = subject_dir / "ts_write_1_sensor.csv"
    write_sensor(sensor, n=3)
    frame = pd.read_csv(sensor)
    frame.loc[1, "Uc2"] = 2.0
    frame.to_csv(sensor, index=False)

    sample = load_udeco_sample(sensor, data_root=tmp_path)

    assert np.isfinite(sample.udeco2).all()
    assert sample.qc["ud_denominator_near_zero"].tolist() == [0, 1, 0]


def test_discovery_requires_strict_sensor_hr_pairs(tmp_path: Path) -> None:
    subject_dir = tmp_path / "ts"
    subject_dir.mkdir()
    write_sensor(subject_dir / "ts_write_1_sensor.csv")
    write_hr(subject_dir / "ts_write_1_HRdata.csv")
    write_sensor(subject_dir / "ts_run_2_sensor.csv")
    write_sensor(subject_dir / "other_rope_1_sensor.csv")
    (subject_dir / "notes.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    pairs, issues = discover_subject_samples(tmp_path, "ts")

    assert [pair.sample_name for pair in pairs] == ["ts_write_1"]
    categories = {issue.file_name: issue.category for issue in issues}
    assert categories["ts_run_2_sensor.csv"] == "missing_pair"
    assert categories["other_rope_1_sensor.csv"] == "subject_mismatch"
    assert categories["notes.csv"] == "invalid_name"


def test_missing_calibration_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ck.mat"):
        load_calibration(tmp_path)


def test_robust_sigma_uses_epsilon_for_constant_signal() -> None:
    assert robust_sigma(np.ones(20), epsilon=1e-7) == pytest.approx(1e-7)


def test_motion_detection_finds_rest_motion_recovery() -> None:
    fs = 100
    time = np.arange(3000, dtype=float) / fs
    rng = np.random.default_rng(12)
    accx = rng.normal(0.0, 0.01, time.size)
    accy = rng.normal(0.0, 0.01, time.size)
    accz = rng.normal(0.0, 0.01, time.size)
    motion = (time >= 10.0) & (time < 20.0)
    accx[motion] += np.sin(2 * np.pi * 2.0 * time[motion])
    config = AnalysisConfig(
        motion_calibration_sec=5.0,
        motion_window_sec=1.0,
        motion_transition_windows=2,
    )

    result = detect_motion_stage(accx, accy, accz, fs=fs, config=config)

    assert result["status"] == "ok"
    assert result["motion_start_s"] == pytest.approx(10.0, abs=1.0)
    assert result["motion_end_s"] == pytest.approx(20.0, abs=1.0)


def test_shared_module_exposes_data_and_motion_without_legacy_classifier():
    from dataclasses import fields
    import udeco_analysis as shared

    for name in ('classify_udeco', 'analyze_sample', 'analyze_all_subjects', 'export_excel',
                 'export_batch_figures', 'estimate_dynamic_m'):
        assert not hasattr(shared, name), name
    assert {f.name for f in fields(shared.AnalysisConfig)} == {
        'epsilon', 'motion_calibration_sec', 'motion_window_sec', 'motion_step_sec',
        'motion_threshold_sigma', 'motion_transition_windows',
    }
    for name in ('load_udeco_sample', 'load_calibration', 'discover_subject_samples',
                 'detect_motion_stage', '_repair_pydisplay_sequence'):
        assert callable(getattr(shared, name))


def test_shared_motion_config_rejects_invalid_parameters():
    for overrides in ({'epsilon': float('nan')}, {'motion_calibration_sec': 0},
                      {'motion_window_sec': -1}, {'motion_step_sec': 0},
                      {'motion_threshold_sigma': float('inf')},
                      {'motion_transition_windows': 0}, {'motion_transition_windows': 1.5},
                      {'motion_transition_windows': True}):
        with pytest.raises(ValueError):
            AnalysisConfig(**overrides)
