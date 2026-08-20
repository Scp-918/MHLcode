"""Udeco2/Udeco3 自适应状态分析工具。

本模块独立复现 change6 仓库中与 Pydisplay CSV、共享 ``ck.mat`` 标定和
100 Hz 时间轴相关的最小读取契约，不依赖或修改参考仓库。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


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
    """状态识别参数；所有时间参数均使用秒。"""

    median_filter_sec: float = 0.03
    initial_m_sec: float = 30.0
    change_window_sec: float = 0.03
    change_threshold_sigma: float = 5.0
    change_peak_merge_sec: float = 0.04
    min_platform_sec: float = 0.06
    state_separation_sigma: float = 4.0
    m_anchor_sigma: float = 2.5
    m_baseline_smooth_sec: float = 10.0
    m_baseline_max_iter: int = 5
    enter_sigma: float = 4.0
    exit_sigma: float = 2.0
    use_gmm: bool = False
    epsilon: float = 1e-9
    motion_calibration_sec: float = 30.0
    motion_window_sec: float = 8.0
    motion_step_sec: float = 1.0
    motion_threshold_sigma: float = 3.0
    motion_transition_windows: int = 6

    def __post_init__(self) -> None:
        positive = {
            "median_filter_sec": self.median_filter_sec,
            "initial_m_sec": self.initial_m_sec,
            "change_window_sec": self.change_window_sec,
            "change_threshold_sigma": self.change_threshold_sigma,
            "change_peak_merge_sec": self.change_peak_merge_sec,
            "min_platform_sec": self.min_platform_sec,
            "state_separation_sigma": self.state_separation_sigma,
            "m_anchor_sigma": self.m_anchor_sigma,
            "m_baseline_smooth_sec": self.m_baseline_smooth_sec,
            "enter_sigma": self.enter_sigma,
            "exit_sigma": self.exit_sigma,
            "epsilon": self.epsilon,
        }
        invalid = [name for name, value in positive.items() if not np.isfinite(value) or value <= 0]
        if invalid:
            raise ValueError(f"Config values must be finite and positive: {', '.join(invalid)}")
        if self.enter_sigma <= self.exit_sigma:
            raise ValueError("enter_sigma must be greater than exit_sigma")
        if int(self.m_baseline_max_iter) < 1:
            raise ValueError("m_baseline_max_iter must be at least 1")


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


def estimate_initial_m(
    signal: np.ndarray,
    *,
    fs: int,
    initial_m_sec: float,
    epsilon: float = 1e-9,
) -> tuple[float, float, int]:
    """由记录开头估计 ``(M0, sigma0, 使用样本数)``。"""

    arr = np.asarray(signal, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("signal must contain finite values")
    n_initial = min(arr.size, max(1, int(round(float(initial_m_sec) * int(fs)))))
    initial = arr[:n_initial]
    m0 = float(np.median(initial))
    return m0, robust_sigma(initial, epsilon=epsilon), n_initial


def detect_change_points(
    signal: np.ndarray,
    *,
    fs: int,
    sigma0: float,
    config: AnalysisConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """用短时左右窗口中位数差检测并合并平台变化点。"""

    arr = np.asarray(signal, dtype=float)
    n = arr.size
    window = max(1, int(round(config.change_window_sec * int(fs))))
    score = np.zeros(n, dtype=float)
    scale = max(float(sigma0), config.epsilon)
    if n < 2 * window + 1:
        return np.asarray([], dtype=int), score
    for index in range(window, n - window):
        left = float(np.median(arr[index - window : index]))
        right = float(np.median(arr[index : index + window]))
        score[index] = abs(right - left) / scale
    peaks, _ = find_peaks(score, height=config.change_threshold_sigma)
    if peaks.size == 0:
        return np.asarray([], dtype=int), score

    # 将得分峰细化到局部最大相邻跃变处，使 segment 边界贴近真实阶跃。
    refined: list[int] = []
    local_delta = np.abs(np.diff(arr))
    for peak in peaks:
        lo = max(0, int(peak) - window)
        hi = min(local_delta.size, int(peak) + window)
        boundary = int(np.argmax(local_delta[lo:hi]) + lo + 1) if hi > lo else int(peak)
        refined.append(boundary)
    candidates = sorted(set(index for index in refined if 0 < index < n))
    merge_samples = max(1, int(round(config.change_peak_merge_sec * int(fs))))
    groups: list[list[int]] = []
    for index in candidates:
        if not groups or index - groups[-1][-1] > merge_samples:
            groups.append([index])
        else:
            groups[-1].append(index)
    merged = [max(group, key=lambda idx: score[min(max(idx, 0), n - 1)]) for group in groups]
    return np.asarray(merged, dtype=int), score


def build_segments(
    raw: np.ndarray,
    filtered: np.ndarray,
    time: np.ndarray,
    change_points: np.ndarray,
    *,
    fs: int,
) -> pd.DataFrame:
    """以变化点为自然边界，计算每个平台的稳健与常规统计量。"""

    raw_arr = np.asarray(raw, dtype=float)
    filt_arr = np.asarray(filtered, dtype=float)
    time_arr = np.asarray(time, dtype=float)
    if not (raw_arr.size == filt_arr.size == time_arr.size):
        raise ValueError("raw, filtered and time must have equal length")
    boundaries = [0, *sorted(set(int(x) for x in change_points if 0 < int(x) < raw_arr.size)), raw_arr.size]
    records: list[dict[str, Any]] = []
    for segment_id, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        values = raw_arr[start:end]
        filtered_values = filt_arr[start:end]
        median = float(np.median(values))
        records.append(
            {
                "segment_id": segment_id,
                "start_idx": start,
                "end_idx": end,
                "start_time": float(time_arr[start]),
                "end_time": float(time_arr[end - 1] + 1.0 / fs),
                "duration": float((end - start) / fs),
                "median": median,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "MAD": float(np.median(np.abs(values - median))),
                "filtered_median": float(np.median(filtered_values)),
            }
        )
    return pd.DataFrame.from_records(records)


def classify_segments(
    segments: pd.DataFrame,
    *,
    raw: np.ndarray,
    m0: float,
    sigma0: float,
    initial_n: int,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """用 M0 锚点给自然 segment 生成第一次 M/H/L 粗分类。"""

    result = segments.copy()
    threshold = config.state_separation_sigma * max(float(sigma0), config.epsilon)
    labels: list[str] = []
    raw_arr = np.asarray(raw, dtype=float)
    for row in result.itertuples(index=False):
        overlap_end = min(int(row.end_idx), int(initial_n))
        initial_center = (
            float(np.median(raw_arr[int(row.start_idx) : overlap_end]))
            if int(row.start_idx) < overlap_end
            else float("nan")
        )
        center = initial_center if np.isfinite(initial_center) else float(row.median)
        distance = center - float(m0)
        if abs(distance) <= threshold:
            labels.append("M")
        else:
            labels.append("H" if distance > 0 else "L")
    if config.use_gmm and len(result) >= 2:
        labels = _gmm_assisted_labels(result, labels, m0=m0, sigma0=sigma0, config=config)
    result["initial_label"] = labels
    return result


def estimate_dynamic_m(
    filtered: np.ndarray,
    segments: pd.DataFrame,
    *,
    fs: int,
    sigma0: float,
    config: AnalysisConfig,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """迭代扩展可信 M segment，并估计缓慢变化的 M(t)。"""

    arr = np.asarray(filtered, dtype=float)
    work = segments.copy()
    trusted_ids = set(work.loc[work["initial_label"] == "M", "segment_id"].astype(int))
    if not trusted_ids:
        trusted_ids.add(int(work.iloc[0]["segment_id"]))
    m_t = np.full(arr.size, float(np.median(arr)), dtype=float)
    for _ in range(int(config.m_baseline_max_iter)):
        trusted_mask = _segment_sample_mask(arr.size, work, trusted_ids)
        m_t = _baseline_from_mask(arr, trusted_mask, fs=fs, smooth_sec=config.m_baseline_smooth_sec)
        trusted_residual = arr[trusted_mask] - m_t[trusted_mask]
        scale = max(float(sigma0), robust_sigma(trusted_residual, epsilon=config.epsilon))
        residual_medians = []
        for row in work.itertuples(index=False):
            start, end = int(row.start_idx), int(row.end_idx)
            residual_medians.append(float(np.median(arr[start:end] - m_t[start:end])))
        work["residual_median"] = residual_medians
        expanded = set(
            work.loc[
                work["residual_median"].abs() <= config.m_anchor_sigma * scale,
                "segment_id",
            ].astype(int)
        ) | trusted_ids
        if expanded == trusted_ids:
            break
        trusted_ids = expanded
    trusted_mask = _segment_sample_mask(arr.size, work, trusted_ids)
    m_t = _baseline_from_mask(arr, trusted_mask, fs=fs, smooth_sec=config.m_baseline_smooth_sec)
    work["trusted_m"] = work["segment_id"].astype(int).isin(trusted_ids)
    work["residual_median"] = [
        float(np.median(arr[int(row.start_idx) : int(row.end_idx)] - m_t[int(row.start_idx) : int(row.end_idx)]))
        for row in work.itertuples(index=False)
    ]
    return m_t, trusted_mask, work


def apply_hysteresis(
    residual_medians: np.ndarray,
    durations: np.ndarray,
    *,
    sigma_m: float,
    config: AnalysisConfig,
) -> tuple[np.ndarray, list[str]]:
    """在 segment 层执行进入/退出阈值不同的 H/M/L 滞回。"""

    residuals = np.asarray(residual_medians, dtype=float)
    durations_arr = np.asarray(durations, dtype=float)
    enter = config.enter_sigma * max(float(sigma_m), config.epsilon)
    exit_threshold = config.exit_sigma * max(float(sigma_m), config.epsilon)
    state = "M"
    labels: list[str] = []
    qc_flags: list[str] = []
    for residual, duration in zip(residuals, durations_arr):
        allow_new_state = duration + config.epsilon >= config.min_platform_sec
        previous = state
        if state == "M":
            if allow_new_state and residual > enter:
                state = "H"
            elif allow_new_state and residual < -enter:
                state = "L"
        elif state == "H":
            if allow_new_state and residual < -enter:
                state = "L"
            elif residual < exit_threshold:
                state = "M"
        else:  # L
            if allow_new_state and residual > enter:
                state = "H"
            elif residual > -exit_threshold:
                state = "M"
        if previous == "H" and state == "L":
            qc_flags.append("direct_H_to_L")
        elif previous == "L" and state == "H":
            qc_flags.append("direct_L_to_H")
        labels.append(state)
    return np.asarray(labels, dtype=object), list(dict.fromkeys(qc_flags))


def classify_udeco(
    raw: np.ndarray,
    *,
    time: np.ndarray | None = None,
    fs: int = FS_HZ,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    """完整执行单路 Udeco 的变化点、动态 M 与滞回分类。"""

    config = config or AnalysisConfig()
    raw_arr = np.asarray(raw, dtype=float).ravel()
    if raw_arr.size < 3 or not np.isfinite(raw_arr).all():
        raise ValueError("raw must contain at least three finite samples")
    if int(fs) <= 0:
        raise ValueError("fs must be positive")
    time_arr = np.arange(raw_arr.size, dtype=float) / float(fs) if time is None else np.asarray(time, dtype=float)
    if time_arr.shape != raw_arr.shape or not np.isfinite(time_arr).all():
        raise ValueError("time must be finite and have the same shape as raw")
    filter_samples = _odd_samples(config.median_filter_sec, fs)
    filtered = median_filter(raw_arr, size=filter_samples, mode="nearest")
    m0, sigma0, initial_n = estimate_initial_m(
        filtered,
        fs=fs,
        initial_m_sec=config.initial_m_sec,
        epsilon=config.epsilon,
    )
    change_points, score = detect_change_points(filtered, fs=fs, sigma0=sigma0, config=config)
    segments = build_segments(raw_arr, filtered, time_arr, change_points, fs=fs)
    segments = classify_segments(
        segments,
        raw=raw_arr,
        m0=m0,
        sigma0=sigma0,
        initial_n=initial_n,
        config=config,
    )
    m_t, trusted_mask, segments = estimate_dynamic_m(
        filtered,
        segments,
        fs=fs,
        sigma0=sigma0,
        config=config,
    )
    residual = filtered - m_t
    sigma_m = max(sigma0, robust_sigma(residual[trusted_mask], epsilon=config.epsilon))
    segment_labels, qc_flags = apply_hysteresis(
        segments["residual_median"].to_numpy(dtype=float),
        segments["duration"].to_numpy(dtype=float),
        sigma_m=sigma_m,
        config=config,
    )
    segments["final_label"] = segment_labels
    segments["platform_ok"] = segments["duration"] + config.epsilon >= config.min_platform_sec
    labels = np.full(raw_arr.size, "M", dtype=object)
    for row in segments.itertuples(index=False):
        labels[int(row.start_idx) : int(row.end_idx)] = str(row.final_label)
    if initial_n < int(round(config.initial_m_sec * fs)):
        qc_flags.append("initial_window_short")
    statistics = _state_statistics(raw_arr, labels, filtered - m_t, fs=fs, m0=m0, sigma0=sigma0, sigma_m=sigma_m)
    statistics["QC_flag"] = ";".join(dict.fromkeys(qc_flags)) if qc_flags else "OK"
    return {
        "time": time_arr,
        "raw": raw_arr,
        "filtered": filtered,
        "labels": labels,
        "M_t": m_t,
        "residual": residual,
        "change_points": change_points,
        "segments": segments,
        "statistics": statistics,
        "debug": {
            "change_score": score,
            "sigma_m": sigma_m,
            "enter_threshold": config.enter_sigma * sigma_m,
            "exit_threshold": config.exit_sigma * sigma_m,
            "filter_samples": filter_samples,
            "trusted_m_mask": trusted_mask,
        },
    }


def _state_statistics(
    raw: np.ndarray,
    labels: np.ndarray,
    residual: np.ndarray,
    *,
    fs: int,
    m0: float,
    sigma0: float,
    sigma_m: float,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    present: list[str] = []
    non_m_separation: list[float] = []
    for state in ("H", "M", "L"):
        mask = labels == state
        if mask.any():
            present.append(state)
            values = raw[mask]
            stats[f"{state}_mean"] = float(np.mean(values))
            stats[f"{state}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            stats[f"{state}_total_duration"] = float(mask.sum() / fs)
            if state != "M":
                non_m_separation.append(abs(float(np.median(residual[mask]))) / max(sigma_m, np.finfo(float).eps))
        else:
            stats[f"{state}_mean"] = float("nan")
            stats[f"{state}_std"] = float("nan")
            stats[f"{state}_total_duration"] = float("nan")
    stats.update(
        {
            "n_detected_states": len(present),
            "transition_count": int(np.sum(labels[1:] != labels[:-1])),
            "M0": float(m0),
            "sigma0": float(sigma0),
            "sigma_M": float(sigma_m),
            "state_separation": min(non_m_separation) if non_m_separation else float("nan"),
        }
    )
    return stats


def _baseline_from_mask(signal: np.ndarray, mask: np.ndarray, *, fs: int, smooth_sec: float) -> np.ndarray:
    source = pd.Series(np.where(mask, signal, np.nan), dtype=float)
    filled = source.interpolate(limit_direction="both")
    if not filled.notna().any():
        filled[:] = float(np.median(signal))
    window = _odd_samples(smooth_sec, fs)
    return filled.rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)


def _segment_sample_mask(length: int, segments: pd.DataFrame, trusted_ids: set[int]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for row in segments.itertuples(index=False):
        if int(row.segment_id) in trusted_ids:
            mask[int(row.start_idx) : int(row.end_idx)] = True
    return mask


def _odd_samples(seconds: float, fs: int) -> int:
    samples = max(1, int(round(float(seconds) * int(fs))))
    return samples if samples % 2 == 1 else samples + 1


def _gmm_assisted_labels(
    segments: pd.DataFrame,
    labels: list[str],
    *,
    m0: float,
    sigma0: float,
    config: AnalysisConfig,
) -> list[str]:
    """可选 GMM/BIC 辅助；默认关闭且不改变主流程依赖。"""

    from sklearn.mixture import GaussianMixture

    values = segments["median"].to_numpy(dtype=float).reshape(-1, 1)
    models = []
    for components in range(1, min(3, len(values)) + 1):
        model = GaussianMixture(n_components=components, random_state=0, reg_covar=config.epsilon)
        model.fit(values)
        models.append((model.bic(values), model))
    model = min(models, key=lambda item: item[0])[1]
    centers = model.means_.ravel()
    m_component = int(np.argmin(np.abs(centers - m0)))
    assignments = model.predict(values)
    threshold = config.state_separation_sigma * max(sigma0, config.epsilon)
    assisted: list[str] = []
    for existing, component in zip(labels, assignments):
        center = float(centers[component])
        if component == m_component or abs(center - m0) <= threshold:
            assisted.append("M")
        else:
            assisted.append("H" if center > m0 else "L")
    return assisted


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
