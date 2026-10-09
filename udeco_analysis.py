"""Udeco 共享数据读取、标定、序号修复和 ACC 运动段定位。

Uh–ROI 状态识别由 udeco_uh_roi_edge_detection 实现；本模块不再提供旧 H/M/L 分类。
"""


from __future__ import annotations


import re


from dataclasses import dataclass


from pathlib import Path


from typing import Any


import numpy as np


import pandas as pd


from scipy.io import loadmat


from scipy.signal import butter, filtfilt


FS_HZ = 100


LEGAL_MOTION_TYPES = ("write", "gripper", "run", "rope")


PYDISPLAY_SENSOR_COLUMNS = (
    "frame_seq",
    "absolute_seq_u64",
    "segment_id",
    "sample_seq",
    "PPG_G",
    "PPG_R",
    "PPG_IR",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "GYRO_X",
    "GYRO_Y",
    "GYRO_Z",
    "Uh1",
    "Uh2",
    "Uh3",
    "Uh4",
    "Uc1",
    "Uc2",
    "Uc3",
    "Uc4",
    "UD1",
    "UD2",
    "parser_valid",
)


_FILE_RE = re.compile(
    r"^(?P<subject>.+)_(?P<motion>write|gripper|run|rope)_"
    r"(?P<index>\d+)_(?P<kind>sensor|HRdata)\.csv$"
)


@dataclass(frozen=True)
class Calibration:
    """Udeco 标定公式使用的通道 2/3 系数。"""

    source_path: Path
    c2: float
    c3: float


@dataclass(frozen=True)
class SamplePair:
    """严格配对的一份传感器和参考心率文件。"""

    sample_name: str
    subject: str
    motion_type: str
    motion_index: int
    sensor_csv: Path
    hr_csv: Path


@dataclass(frozen=True)
class DiscoveryIssue:
    """不能进入批处理的文件及其原因。"""

    file_name: str
    file_path: Path
    category: str
    reason: str


@dataclass
class LoadedSample:
    """经过序号修复、但未进行状态识别滤波的单样本数据。"""

    sample_name: str
    subject: str
    fs: int
    time_s: np.ndarray
    udeco2: np.ndarray
    udeco3: np.ndarray
    accx: np.ndarray
    accy: np.ndarray
    accz: np.ndarray
    qc: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AnalysisConfig:
    """共享 ACC 运动段定位参数；保留名称以兼容现有调用。"""

    epsilon: float = 1e-9
    motion_calibration_sec: float = 30.0
    motion_window_sec: float = 8.0
    motion_step_sec: float = 1.0
    motion_threshold_sigma: float = 3.0
    motion_transition_windows: int = 6

    def __post_init__(self) -> None:
        positive = {
            "epsilon": self.epsilon,
            "motion_calibration_sec": self.motion_calibration_sec,
            "motion_window_sec": self.motion_window_sec,
            "motion_step_sec": self.motion_step_sec,
            "motion_threshold_sigma": self.motion_threshold_sigma,
        }
        invalid = [name for name, value in positive.items() if not np.isfinite(value) or value <= 0]
        if invalid:
            raise ValueError(f"Config values must be finite and positive: {', '.join(invalid)}")
        if isinstance(self.motion_transition_windows, bool) or not isinstance(self.motion_transition_windows, int) or self.motion_transition_windows < 1:
            raise ValueError("motion_transition_windows must be a positive integer")


def load_calibration(data_root: str | Path) -> Calibration:
    """读取并验证 ``DATA_ROOT/ck.mat`` 中的 c2、c3。"""

    path = Path(data_root).resolve() / "ck.mat"
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    try:
        payload = loadmat(path, simplify_cells=True)
    except Exception as exc:  # pragma: no cover - scipy 提供具体底层错误
        raise ValueError(f"Cannot parse calibration MAT file {path}: {exc}") from exc
    c_result = _cell_sequence(payload, "cResult", minimum=3)
    c2_entry = _mat_struct(c_result[1], "cResult{2}")
    c3_entry = _mat_struct(c_result[2], "cResult{3}")
    _expect_integer(c2_entry, "channel", 2, "cResult{2}.channel")
    _expect_integer(c3_entry, "channel", 3, "cResult{3}.channel")
    return Calibration(
        source_path=path,
        c2=_finite_scalar(c2_entry, "c", "cResult{2}.c"),
        c3=_finite_scalar(c3_entry, "c", "cResult{3}.c"),
    )


def discover_subject_samples(
    data_root: str | Path,
    subject: str,
) -> tuple[list[SamplePair], list[DiscoveryIssue]]:
    """按 change6 的严格文件名契约发现一个受试者的成对样本。"""

    subject_dir = Path(data_root) / subject
    if not subject_dir.is_dir():
        raise NotADirectoryError(f"Subject directory not found: {subject_dir}")
    accepted: dict[tuple[str, int, str], list[Path]] = {}
    issues: list[DiscoveryIssue] = []
    for path in sorted(subject_dir.glob("*.csv"), key=lambda item: item.name):
        match = _FILE_RE.fullmatch(path.name)
        if match is None:
            issues.append(_issue(path, "invalid_name", "filename does not match the subject data contract"))
            continue
        if match.group("subject") != subject:
            issues.append(_issue(path, "subject_mismatch", "filename subject differs from directory"))
            continue
        if not _has_data_row(path):
            issues.append(_issue(path, "empty_data", "CSV contains no data rows"))
            continue
        key = (match.group("motion"), int(match.group("index")), match.group("kind"))
        accepted.setdefault(key, []).append(path)

    duplicate_keys = {key for key, paths in accepted.items() if len(paths) > 1}
    for key in sorted(duplicate_keys):
        for path in accepted[key]:
            issues.append(_issue(path, "duplicate", "duplicate normalized motion/index/kind"))

    order = {motion: index for index, motion in enumerate(LEGAL_MOTION_TYPES)}
    sample_keys = sorted(
        {(motion, index) for motion, index, _ in accepted},
        key=lambda item: (order[item[0]], item[1]),
    )
    pairs: list[SamplePair] = []
    for motion, index in sample_keys:
        sensor_key = (motion, index, "sensor")
        hr_key = (motion, index, "HRdata")
        sensors = [] if sensor_key in duplicate_keys else accepted.get(sensor_key, [])
        hrs = [] if hr_key in duplicate_keys else accepted.get(hr_key, [])
        if len(sensors) == 1 and len(hrs) == 1:
            sample_name = f"{subject}_{motion}_{index}"
            pairs.append(SamplePair(sample_name, subject, motion, index, sensors[0], hrs[0]))
        else:
            if len(sensors) == 1:
                issues.append(_issue(sensors[0], "missing_pair", "missing matching HRdata CSV"))
            if len(hrs) == 1:
                issues.append(_issue(hrs[0], "missing_pair", "missing matching sensor CSV"))
    return pairs, sorted(issues, key=lambda item: item.file_name)


def load_udeco_sample(
    sensor_csv: str | Path,
    *,
    data_root: str | Path,
    fs: int = FS_HZ,
    epsilon: float = 1e-9,
) -> LoadedSample:
    """读取一份 Pydisplay CSV，修复序号并计算 Udeco2/Udeco3。"""

    path = Path(sensor_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Sensor CSV not found: {path}")
    if int(fs) <= 0:
        raise ValueError("fs must be positive")
    calibration = load_calibration(data_root)
    raw = pd.read_csv(path)
    missing = [column for column in PYDISPLAY_SENSOR_COLUMNS if column not in raw.columns]
    if missing:
        raise KeyError(f"Missing required Pydisplay columns: {', '.join(missing)}")
    frame, qc, metadata = _repair_pydisplay_sequence(raw)

    numeric_columns = ("Uh2", "Uh3", "Uc2", "Uc3", "ACC_X", "ACC_Y", "ACC_Z")
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).any():
            raise ValueError(f"Required channel {column!r} has no finite values")
        frame[column] = values.interpolate(limit_direction="both")

    den2 = calibration.c2 - frame["Uc2"].to_numpy(dtype=float)
    den3 = calibration.c3 - frame["Uc3"].to_numpy(dtype=float)
    near2 = np.isfinite(den2) & (np.abs(den2) <= float(epsilon))
    near3 = np.isfinite(den3) & (np.abs(den3) <= float(epsilon))
    safe2 = den2.copy()
    safe3 = den3.copy()
    safe2[near2] = np.nan
    safe3[near3] = np.nan
    udeco2 = (frame["Uh2"].to_numpy(dtype=float) - frame["Uc2"].to_numpy(dtype=float)) / safe2
    udeco3 = (frame["Uh3"].to_numpy(dtype=float) - frame["Uc3"].to_numpy(dtype=float)) / safe3
    udeco2 = _interpolate_finite(udeco2, "Udeco2")
    udeco3 = _interpolate_finite(udeco3, "Udeco3")
    qc = qc.copy()
    qc["ud_denominator_near_zero"] = (near2 | near3).astype(int)

    sample_name = path.name.removesuffix("_sensor.csv")
    match = _FILE_RE.fullmatch(path.name)
    subject = match.group("subject") if match else path.parent.name
    metadata.update(
        {
            "sensor_csv": str(path.resolve()),
            "calibration_path": str(calibration.source_path),
            "calibration_c2": calibration.c2,
            "calibration_c3": calibration.c3,
            "fs_hz": int(fs),
            "udeco_mapping": "Udeco2=ud1(Uh2/Uc2), Udeco3=ud2(Uh3/Uc3)",
        }
    )
    n = len(frame)
    return LoadedSample(
        sample_name=sample_name,
        subject=subject,
        fs=int(fs),
        time_s=np.arange(n, dtype=float) / float(fs),
        udeco2=udeco2,
        udeco3=udeco3,
        accx=frame["ACC_X"].to_numpy(dtype=float),
        accy=frame["ACC_Y"].to_numpy(dtype=float),
        accz=frame["ACC_Z"].to_numpy(dtype=float),
        qc=qc,
        metadata=metadata,
    )


def robust_sigma(values: np.ndarray, *, epsilon: float = 1e-9) -> float:
    """用 ``1.4826 × MAD`` 估计噪声尺度，并提供数值下限。"""

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(epsilon)
    center = float(np.median(arr))
    mad = float(np.median(np.abs(arr - center)))
    return max(float(epsilon), 1.4826 * mad)


def detect_motion_stage(
    accx: np.ndarray,
    accy: np.ndarray,
    accz: np.ndarray,
    *,
    fs: int = FS_HZ,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    """沿用参考仓库的 ACC 合模长 STD 规则识别运动起止。"""

    config = config or AnalysisConfig()
    axes = [_safe_bandpass(np.asarray(axis, dtype=float), fs, 0.5, 10.0) for axis in (accx, accy, accz)]
    if len({axis.size for axis in axes}) != 1 or axes[0].size == 0:
        return _motion_error("ACC channels must have equal non-zero length")
    magnitude = np.sqrt(sum(axis**2 for axis in axes))
    calibration_n = min(magnitude.size, int(round(config.motion_calibration_sec * fs)))
    if calibration_n < 2:
        return _motion_error("not enough samples for motion calibration")
    baseline_std = float(np.std(magnitude[:calibration_n], ddof=1))
    threshold = config.motion_threshold_sigma * max(baseline_std, config.epsilon)
    window_n = max(2, int(round(config.motion_window_sec * fs)))
    step_n = max(1, int(round(config.motion_step_sec * fs)))
    if magnitude.size < window_n:
        return _motion_error("signal shorter than one motion window", threshold=threshold)
    starts = np.arange(0, magnitude.size - window_n + 1, step_n, dtype=int)
    window_std = np.asarray(
        [np.std(magnitude[start : start + window_n], ddof=1) for start in starts],
        dtype=float,
    )
    flags = window_std > threshold
    run = int(config.motion_transition_windows)
    start_index = _find_boolean_transition(flags, before=False, after=True, run=run, start_at=0)
    if start_index is None:
        return _motion_error(
            f"cannot find {run} rest windows followed by {run} motion windows",
            threshold=threshold,
            starts=starts / fs,
            window_std=window_std,
            flags=flags,
        )
    end_index = _find_boolean_transition(flags, before=True, after=False, run=run, start_at=start_index)
    if end_index is None:
        return _motion_error(
            f"cannot find {run} motion windows followed by {run} rest windows",
            threshold=threshold,
            starts=starts / fs,
            window_std=window_std,
            flags=flags,
        )
    return {
        "status": "ok",
        "reason": "",
        "motion_start_s": float(starts[start_index] / fs),
        "motion_end_s": float(starts[end_index] / fs),
        "motion_threshold": float(threshold),
        "window_starts_s": starts.astype(float) / fs,
        "window_centers_s": starts.astype(float) / fs + config.motion_window_sec / 2.0,
        "window_std": window_std,
        "motion_flags": flags,
    }


def _safe_bandpass(signal: np.ndarray, fs: int, low_hz: float, high_hz: float) -> np.ndarray:
    """对 ACC 使用零相位带通；短序列安全退化为去均值。"""

    arr = np.asarray(signal, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        return arr
    nyquist = float(fs) / 2.0
    high = min(float(high_hz), 0.45 * float(fs))
    low = max(float(low_hz), 1e-3)
    if arr.size < 8 or not (0 < low < high < nyquist):
        return arr - np.mean(arr)
    b, a = butter(4, [low / nyquist, high / nyquist], btype="bandpass")
    padlen = 3 * max(len(a), len(b))
    if arr.size <= padlen:
        return arr - np.mean(arr)
    try:
        return filtfilt(b, a, arr)
    except ValueError:
        return arr - np.mean(arr)


def _find_boolean_transition(
    flags: np.ndarray,
    *,
    before: bool,
    after: bool,
    run: int,
    start_at: int,
) -> int | None:
    for index in range(max(0, int(start_at)), len(flags) - 2 * int(run) + 1):
        if np.all(flags[index : index + run] == before) and np.all(
            flags[index + run : index + 2 * run] == after
        ):
            return index + run
    return None


def _motion_error(
    reason: str,
    *,
    threshold: float = float("nan"),
    starts: np.ndarray | None = None,
    window_std: np.ndarray | None = None,
    flags: np.ndarray | None = None,
) -> dict[str, Any]:
    starts_arr = np.asarray([] if starts is None else starts, dtype=float)
    return {
        "status": "error",
        "reason": reason,
        "motion_start_s": float("nan"),
        "motion_end_s": float("nan"),
        "motion_threshold": float(threshold),
        "window_starts_s": starts_arr,
        "window_centers_s": starts_arr,
        "window_std": np.asarray([] if window_std is None else window_std, dtype=float),
        "motion_flags": np.asarray([] if flags is None else flags, dtype=bool),
    }


def _repair_pydisplay_sequence(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """过滤无效行，并在同一 segment 内补齐 absolute sequence 缺口。"""

    input_rows = len(raw)
    empty_mask = raw.loc[:, list(PYDISPLAY_SENSOR_COLUMNS)].isna().all(axis=1)
    removed_empty = int(empty_mask.sum())
    work = raw.loc[~empty_mask].copy()
    valid = work["parser_valid"].map(_parser_valid_value).fillna(False).astype(bool)
    removed_invalid = int((~valid).sum())
    work = work.loc[valid].copy()
    if work.empty:
        raise ValueError("Sensor CSV has no usable rows after parser_valid filtering")
    for column in ("frame_seq", "absolute_seq_u64", "segment_id", "sample_seq"):
        values = pd.to_numeric(work[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Sequence column {column!r} must contain finite integers")
        work[column] = values.astype(np.int64)

    output: list[dict[str, Any]] = []
    qc_rows: list[dict[str, int | float]] = []
    previous_segment: int | None = None
    previous_absolute: int | None = None
    previous_frame: int | None = None
    removed_reordered = 0
    inserted_missing = 0
    segment_transitions = 0
    for _, row_series in work.iterrows():
        row = row_series.to_dict()
        segment = int(row["segment_id"])
        absolute = int(row["absolute_seq_u64"])
        frame_seq = int(row["frame_seq"])
        if previous_segment is not None and segment != previous_segment:
            segment_transitions += 1
            previous_absolute = None
            previous_frame = None
        if previous_absolute is not None:
            delta = absolute - previous_absolute
            if delta <= 0:
                removed_reordered += 1
                continue
            gap = delta - 1
            for offset in range(1, delta):
                inserted = {column: np.nan for column in PYDISPLAY_SENSOR_COLUMNS}
                inserted["segment_id"] = segment
                inserted["absolute_seq_u64"] = previous_absolute + offset
                inserted["frame_seq"] = (int(previous_frame) + offset) % 65536
                inserted["parser_valid"] = False
                output.append(inserted)
                qc_rows.append(
                    {
                        "SampleIndex": len(output) - 1,
                        "Seq": previous_absolute + offset,
                        "ValidFlag": 0,
                        "InterpFlag": 1,
                        "GapLen": gap,
                        "MissingBefore": 0,
                    }
                )
                inserted_missing += 1
        else:
            gap = 0
        output.append(row)
        qc_rows.append(
            {
                "SampleIndex": len(output) - 1,
                "Seq": absolute,
                "ValidFlag": 1,
                "InterpFlag": 0,
                "GapLen": 0,
                "MissingBefore": gap,
            }
        )
        previous_segment = segment
        previous_absolute = absolute
        previous_frame = frame_seq
    frame = pd.DataFrame(output, columns=PYDISPLAY_SENSOR_COLUMNS)
    qc = pd.DataFrame(qc_rows)
    metadata = {
        "input_rows": input_rows,
        "output_rows": len(frame),
        "removed_empty_rows": removed_empty,
        "removed_parser_invalid_rows": removed_invalid,
        "removed_duplicate_or_reordered_rows": removed_reordered,
        "inserted_missing_rows": inserted_missing,
        "segment_transitions": segment_transitions,
    }
    return frame, qc, metadata


def _interpolate_finite(values: np.ndarray, name: str) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float)).replace([np.inf, -np.inf], np.nan)
    if not series.notna().any():
        raise ValueError(f"{name} has no finite values")
    result = series.interpolate(limit_direction="both").to_numpy(dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains unrecoverable values")
    return result


def _parser_valid_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return float(value) == 1.0
    return str(value).strip().lower() in {"true", "1"}


def _cell_sequence(payload: dict[str, Any], name: str, *, minimum: int) -> list[Any]:
    if name not in payload:
        raise ValueError(f"Calibration MAT missing field {name}")
    value = payload[name]
    items = list(value.ravel()) if isinstance(value, np.ndarray) else list(value) if isinstance(value, (list, tuple)) else [value]
    if len(items) < minimum:
        raise ValueError(f"Calibration field {name} must contain at least {minimum} cells")
    return items


def _mat_struct(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a MATLAB struct")
    return value


def _finite_scalar(entry: dict[str, Any], field: str, label: str) -> float:
    if field not in entry:
        raise ValueError(f"Calibration field missing: {label}")
    value = np.asarray(entry[field])
    if value.size != 1:
        raise ValueError(f"{label} must be scalar")
    scalar = float(value.reshape(-1)[0])
    if not np.isfinite(scalar):
        raise ValueError(f"{label} must be finite")
    return scalar


def _expect_integer(entry: dict[str, Any], field: str, expected: int, label: str) -> None:
    value = _finite_scalar(entry, field, label)
    if value != expected:
        raise ValueError(f"{label} must equal {expected}, got {value:g}")


def _has_data_row(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            next(handle, None)
            return any(line.strip() for line in handle)
    except OSError:
        return False


def _issue(path: Path, category: str, reason: str) -> DiscoveryIssue:
    return DiscoveryIssue(path.name, path, category, reason)
