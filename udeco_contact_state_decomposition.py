"""接触状态分解的第一阶段：稳健预处理、试探趋势与转换候选。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.ndimage import median_filter
from statsmodels.nonparametric.smoothers_lowess import lowess


@dataclass(frozen=True)
class ContactStateConfig:
    """接触状态分析各阶段共用的不可变配置。"""

    hampel_window_s: float = 0.15
    hampel_n_sigma: float = 3.5
    pilot_lowess_window_s: float = 3.0
    pilot_lowess_robust_iterations: int = 2
    candidate_scales: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
    candidate_z_threshold: float = 4.0
    candidate_merge_s: float = 0.25
    refinement_context_s: float = 1.5
    detach_peak_slope_z: float = 3.0
    attach_min_monotonicity: float = 0.70
    initial_contact_s: float = 30.0
    state_min_dwell_s: float = 1.0
    state_transition_confidence: float = 0.50
    downstream_min_segment_s: float = 0.25

    def __post_init__(self) -> None:
        """在分析开始前拒绝没有物理时间意义的配置。"""
        positive = (
            self.hampel_window_s,
            self.hampel_n_sigma,
            self.pilot_lowess_window_s,
            self.candidate_z_threshold,
            self.candidate_merge_s,
            self.refinement_context_s,
            self.detach_peak_slope_z,
            self.initial_contact_s,
            self.state_min_dwell_s,
            self.downstream_min_segment_s,
        )
        if any(not isinstance(value, Real) or not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("配置中的时间和阈值必须为有限正数")
        if not isinstance(self.pilot_lowess_robust_iterations, int) or self.pilot_lowess_robust_iterations < 0:
            raise ValueError("LOWESS稳健迭代次数必须为非负整数")
        if not 0 < self.attach_min_monotonicity <= 1:
            raise ValueError("单调性阈值必须在 (0, 1] 内")
        if not 0 < self.state_transition_confidence <= 1:
            raise ValueError("状态转换置信度阈值必须在 (0, 1] 内")
        if not isinstance(self.candidate_scales, tuple) or not self.candidate_scales or any(
            not isinstance(scale, Real) or not np.isfinite(scale) or scale <= 0
            for scale in self.candidate_scales
        ):
            raise ValueError("候选尺度必须是非空的有限正秒数")


def _validated_signal(signal: np.ndarray, fs: float, config: ContactStateConfig) -> np.ndarray:
    """验证公共接口共享的单通道采样输入。"""
    if not isinstance(config, ContactStateConfig):
        raise TypeError("config 必须是 ContactStateConfig")
    if not isinstance(fs, Real) or isinstance(fs, bool) or not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs 必须是有限正数")
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal 必须是非空一维数组")
    if not np.isfinite(values).all():
        raise ValueError("signal 不能包含 NaN 或无穷值")
    return values


def _mad(values: np.ndarray) -> float:
    """返回未缩放的中位数绝对偏差。"""
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def hampel_filter(signal: np.ndarray, fs: float, config: ContactStateConfig) -> np.ndarray:
    """以局部中位数和 MAD 替换单点脉冲异常值。"""
    values = _validated_signal(signal, fs, config)
    radius = max(1, int(round(config.hampel_window_s * fs / 2.0)))
    clean = values.copy()
    for index in range(values.size):
        left = max(0, index - radius)
        right = min(values.size, index + radius + 1)
        window = values[left:right]
        median = float(np.median(window))
        scale = 1.4826 * _mad(window)
        deviation = abs(values[index] - median)
        if (scale == 0.0 and deviation > 0.0) or (scale > 0.0 and deviation > config.hampel_n_sigma * scale):
            clean[index] = median
    return clean


def estimate_pilot_trend(clean: np.ndarray, fs: float, config: ContactStateConfig) -> dict:
    """用 3 秒、两次稳健迭代的 LOWESS 构造试探基线和快速残差。"""
    values = _validated_signal(clean, fs, config)
    if values.size < 3:
        raise ValueError("估计试探趋势至少需要三个采样点")
    duration_s = values.size / float(fs)
    fraction = min(1.0, max(3.0 / values.size, config.pilot_lowess_window_s / duration_s))
    sample_time = np.arange(values.size, dtype=float) / float(fs)
    trend = lowess(
        values,
        sample_time,
        frac=fraction,
        it=config.pilot_lowess_robust_iterations,
        return_sorted=False,
    )
    trend = np.asarray(trend, dtype=float)
    return {
        "Bpilot": trend,
        "fast_residual": values - trend,
        "pilot_slope": np.gradient(trend) * float(fs),
    }


def _transition_bounds(
    smooth: np.ndarray,
    center: int,
    direction: str,
    fs: float,
    context_s: float,
) -> tuple[int, int, float]:
    """从稳健平滑残差中以 10–90% 交叉点细化转换边界。"""
    context = max(2, int(round(context_s * fs)))
    guard = max(1, int(round(0.05 * fs)))
    pre = smooth[max(0, center - context) : max(1, center - guard)]
    post = smooth[min(smooth.size - 1, center + guard) : min(smooth.size, center + context)]
    if pre.size == 0 or post.size == 0:
        return max(0, center - 1), min(smooth.size - 1, center + 1), 0.0
    pre_level = float(np.median(pre))
    post_level = float(np.median(post))
    delta = post_level - pre_level
    target_10 = pre_level + 0.1 * delta
    target_90 = pre_level + 0.9 * delta
    left_start = max(0, center - 2 * context)
    right_end = min(smooth.size, center + 2 * context + 1)
    before = smooth[left_start : center + 1]
    after = smooth[center:right_end]
    if direction == "up":
        starts = np.flatnonzero(before <= target_10)
        ends = np.flatnonzero(after >= target_90)
    else:
        starts = np.flatnonzero(before >= target_10)
        ends = np.flatnonzero(after <= target_90)
    start = left_start + int(starts[-1]) if starts.size else max(0, center - 1)
    end = center + int(ends[0]) if ends.size else min(smooth.size - 1, center + 1)
    if end <= start:
        start, end = max(0, center - 1), min(smooth.size - 1, center + 1)
    return start, end, delta


def detect_transition_candidates(
    clean: np.ndarray,
    bpilot: np.ndarray,
    fs: float,
    config: ContactStateConfig,
) -> tuple[list[dict], dict]:
    """以多尺度前后稳健中位数差检测并细化接触转换候选。"""
    values = _validated_signal(clean, fs, config)
    baseline = _validated_signal(bpilot, fs, config)
    if baseline.shape != values.shape:
        raise ValueError("clean 和 bpilot 必须具有相同长度")
    if values.size < 5:
        raise ValueError("候选检测至少需要五个采样点")

    residual = values - baseline
    scores = np.zeros(values.size, dtype=float)
    signed_changes = np.zeros(values.size, dtype=float)
    contributing_scales: list[float] = []
    for scale_s in config.candidate_scales:
        width = max(2, int(round(scale_s * fs)))
        if 2 * width >= values.size:
            continue
        contributing_scales.append(float(scale_s))
        for index in range(width, values.size - width):
            before = residual[index - width : index]
            after = residual[index : index + width]
            change = float(np.median(after) - np.median(before))
            noise = max(_mad(before), _mad(after), np.finfo(float).eps)
            score = abs(change) / (1.4826 * noise)
            if score > scores[index]:
                scores[index] = score
                signed_changes[index] = change

    active = np.flatnonzero(scores >= config.candidate_z_threshold)
    if active.size == 0:
        return [], {"residual": residual, "scores": scores, "scales_s": contributing_scales}

    merge_gap = max(1, int(round(config.candidate_merge_s * fs)))
    groups: list[np.ndarray] = []
    group_start = 0
    for position in range(1, active.size):
        if active[position] - active[position - 1] > merge_gap:
            groups.append(active[group_start:position])
            group_start = position
    groups.append(active[group_start:])

    smooth_size = max(3, int(round(0.05 * fs)) | 1)
    smooth = median_filter(residual, size=smooth_size, mode="nearest")
    slope = np.gradient(smooth) * float(fs)
    candidates: list[dict] = []
    for group in groups:
        center = int(group[np.argmax(scores[group])])
        direction = "up" if signed_changes[center] >= 0 else "down"
        start, end, delta = _transition_bounds(
            smooth, center, direction, fs, config.refinement_context_s
        )
        derivative = slope[start : end + 1]
        peak_slope = float(np.max(np.abs(derivative))) if derivative.size else 0.0
        context = max(2, int(round(config.refinement_context_s * fs)))
        noise_slope = np.concatenate(
            (slope[max(0, center - context) : max(0, start - 1)], slope[min(slope.size, end + 1) : min(slope.size, center + context)])
        )
        peak_slope_z = peak_slope / max(1.4826 * _mad(noise_slope), np.finfo(float).eps) if noise_slope.size else 0.0
        directional_slope = derivative if direction == "up" else -derivative
        monotonicity = float(np.mean(directional_slope >= 0.0)) if directional_slope.size else 0.0
        duration_s = (end - start) / float(fs)
        if duration_s <= 0.20 and peak_slope_z >= config.detach_peak_slope_z:
            candidate_type = "detach"
        elif 0.20 < duration_s <= 3.0 and monotonicity >= config.attach_min_monotonicity:
            candidate_type = "attach"
        else:
            candidate_type = "ambiguous"
        confidence = float(np.clip((scores[center] / config.candidate_z_threshold) * monotonicity / 2.0, 0.0, 1.0))
        candidates.append(
            {
                "start_index": start,
                "end_index": end,
                "start_s": start / float(fs),
                "end_s": end / float(fs),
                "duration_s": duration_s,
                "delta": float(delta),
                "direction": direction,
                "candidate_type": candidate_type,
                "peak_slope": peak_slope,
                "peak_slope_z": float(peak_slope_z),
                "monotonicity": monotonicity,
                "confidence": confidence,
                "accepted": False,
                "rejection_reason": "awaiting_state_inference",
            }
        )
    candidates.sort(key=lambda candidate: candidate["start_index"])
    return candidates, {"residual": residual, "scores": scores, "scales_s": contributing_scales}
