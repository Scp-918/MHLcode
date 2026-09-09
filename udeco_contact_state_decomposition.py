"""基于接触状态的 Udeco 单通道分解算法。"""

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
    candidate_refine_smooth_s: float = 0.20
    detach_peak_slope_z: float = 3.0
    attach_min_monotonicity: float = 0.70
    initial_contact_s: float = 30.0
    state_min_dwell_s: float = 0.5
    state_transition_confidence: float = 0.50
    contact_anchor_window_s: float = 1.0
    max_dynamic_fraction: float = 0.15
    state_pilot_slope_relative_threshold: float = 0.03
    contact_anchor_residual_fraction: float = 0.07
    terminal_min_tail_s: float = 2.0
    downstream_min_segment_s: float = 0.25
    contact_lowess_window_s: float = 3.0
    contact_lowess_robust_iterations: int = 2
    bslow_tau_s: float = 60.0
    bslow_update_rate: float = 1.0
    bslow_stable_edge_threshold: float = 3.5
    bslow_stable_slope_threshold: float = 3e-4
    bslow_stable_offset_threshold: float = 1e-3
    j_smooth_window_s: float = 0.10

    def __post_init__(self) -> None:
        """在分析开始前拒绝没有物理时间意义的配置。"""
        positive = (
            self.hampel_window_s,
            self.hampel_n_sigma,
            self.pilot_lowess_window_s,
            self.candidate_z_threshold,
            self.candidate_merge_s,
            self.refinement_context_s,
            self.candidate_refine_smooth_s,
            self.detach_peak_slope_z,
            self.initial_contact_s,
            self.state_min_dwell_s,
            self.contact_anchor_window_s,
            self.state_pilot_slope_relative_threshold,
            self.contact_anchor_residual_fraction,
            self.terminal_min_tail_s,
            self.downstream_min_segment_s,
            self.contact_lowess_window_s,
            self.bslow_tau_s,
            self.bslow_update_rate,
            self.bslow_stable_edge_threshold,
            self.bslow_stable_slope_threshold,
            self.bslow_stable_offset_threshold,
            self.j_smooth_window_s,
        )
        if any(not isinstance(value, Real) or not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("配置中的时间和阈值必须为有限正数")
        if not isinstance(self.pilot_lowess_robust_iterations, int) or self.pilot_lowess_robust_iterations < 0:
            raise ValueError("LOWESS稳健迭代次数必须为非负整数")
        if not isinstance(self.contact_lowess_robust_iterations, int) or self.contact_lowess_robust_iterations < 0:
            raise ValueError("最终 LOWESS 稳健迭代次数必须为非负整数")
        if not 0 < self.attach_min_monotonicity <= 1:
            raise ValueError("单调性阈值必须在 (0, 1] 内")
        if not 0 < self.state_transition_confidence <= 1:
            raise ValueError("状态转换置信度阈值必须在 (0, 1] 内")
        if not isinstance(self.max_dynamic_fraction, Real) or not np.isfinite(self.max_dynamic_fraction) or not 0 <= self.max_dynamic_fraction <= 1:
            raise ValueError("动态样本占比阈值必须在 [0, 1] 内")
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
    # 局部 MAD 在近乎平坦窗口中可能趋近于零，因此以初始稳定段的通道噪声
    # 作为下限。短合成序列只取前四分之一，避免把测试中的真实事件混入参考。
    reference_end = max(
        5,
        min(values.size, int(round(min(config.initial_contact_s, values.size / fs * 0.25) * fs))),
    )
    reference_noise = _mad(residual[:reference_end])
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
            noise = max(_mad(before), _mad(after), reference_noise, np.finfo(float).eps)
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

    # 候选仍由 fast residual 检出；稍长的中值窗只用于稳定 10%–90% 边界，
    # 避免单个噪声脉冲把慢 attach 的持续时间压缩成一两个采样点。
    smooth_size = max(3, int(round(config.candidate_refine_smooth_s * fs)) | 1)
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


def _candidate_bounds(candidate: dict, size: int) -> tuple[int, int]:
    """返回限制在信号范围内的闭区间候选边界。"""
    try:
        start = int(candidate["start_index"])
        end = int(candidate["end_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("候选必须包含有效的 start_index 和 end_index") from exc
    if start < 0 or end < start or end >= size:
        raise ValueError("候选边界必须位于信号范围内且顺序有效")
    return start, end


def _validate_candidate(candidate: dict, size: int) -> None:
    """只校验状态推断直接依赖的候选字段。"""
    _candidate_bounds(candidate, size)
    if candidate.get("candidate_type") not in {"detach", "attach", "ambiguous"}:
        raise ValueError("候选 candidate_type 必须是 detach、attach 或 ambiguous")
    if candidate.get("direction") not in {"up", "down"}:
        raise ValueError("候选 direction 必须是 up 或 down")
    duration = candidate.get("duration_s")
    duration_valid = (
        isinstance(duration, Real)
        and not isinstance(duration, bool)
        and np.isfinite(duration)
        and duration >= 0
    )
    if not duration_valid:
        raise ValueError("候选 duration_s 必须是有限非负数")
    confidence = candidate.get("confidence")
    confidence_valid = (
        isinstance(confidence, Real)
        and not isinstance(confidence, bool)
        and np.isfinite(confidence)
        and 0 <= confidence <= 1
    )
    if not confidence_valid:
        raise ValueError("候选 confidence 必须是 [0, 1] 内的有限数")


def _pilot_slope_activity(
    bpilot: np.ndarray,
    fs: float,
    reference_end: int,
    config: ContactStateConfig,
) -> np.ndarray:
    """以强制接触参考段的稳健斜率背景标记全段趋势活动。"""
    slope = np.gradient(bpilot) * float(fs)
    reference_slope = slope[:reference_end]
    center = float(np.median(reference_slope))
    scale = 1.4826 * _mad(reference_slope)
    numerical_floor = np.sqrt(np.finfo(float).eps) * max(
        1.0,
        float(np.median(np.abs(bpilot[:reference_end]))) * float(fs),
    )
    relative_floor = config.state_pilot_slope_relative_threshold * float(
        np.median(np.abs(bpilot[:reference_end]))
    )
    threshold = max(3.0 * scale, relative_floor, numerical_floor)
    return np.abs(slope - center) > threshold


def _activity_fraction(activity: np.ndarray) -> float:
    """返回布尔窗口中的动态样本占比；空窗口视作不可靠。"""
    return float(np.mean(activity)) if activity.size else 1.0


def _residual_noise_limit(
    clean: np.ndarray,
    bpilot: np.ndarray,
    reference_end: int,
    config: ContactStateConfig,
) -> float:
    """由强制接触参考段给出 clean-Bpilot 的稳健锚窗噪声上限。"""
    reference = clean[:reference_end] - bpilot[:reference_end]
    reference_scale = 1.4826 * _mad(reference)
    reference_level = max(np.finfo(float).eps, float(np.median(np.abs(clean[:reference_end]))))
    relative_floor = config.contact_anchor_residual_fraction * reference_level
    return max(3.0 * reference_scale, relative_floor)


def _anchor_residual_is_stable(
    clean: np.ndarray,
    bpilot: np.ndarray,
    start: int,
    end: int,
    required_samples: int,
    noise_limit: float,
) -> bool:
    """检查实际重建锚窗的局部残差噪声是否可靠。"""
    residual = clean[start:end] - bpilot[start:end]
    return residual.size == required_samples and 1.4826 * _mad(residual) <= noise_limit


def infer_contact_states(
    clean: np.ndarray,
    bpilot: np.ndarray,
    candidates: list[dict],
    fs: float,
    config: ContactStateConfig,
) -> dict:
    """以保守的 tentative detach→attach 配对推断接触状态。

    状态机允许 tentative detach 与后续 attach 之间夹有噪声候选；只有找到
    方向相反、attach 更慢且区间证据完整的一对边沿，才确认 Non-contact。
    """
    values = _validated_signal(clean, fs, config)
    baseline = _validated_signal(bpilot, fs, config)
    if baseline.shape != values.shape:
        raise ValueError("clean 和 bpilot 必须具有相同长度")
    if values.size < 2:
        raise ValueError("状态推断至少需要两个采样点")
    if not isinstance(candidates, list):
        raise TypeError("candidates 必须是列表")

    updated = [dict(candidate) for candidate in candidates]
    for candidate in updated:
        _validate_candidate(candidate, values.size)
    updated.sort(key=lambda candidate: _candidate_bounds(candidate, values.size)[0])
    contact_mask = np.ones(values.size, dtype=bool)
    noncontact_mask = np.zeros(values.size, dtype=bool)
    transition_mask = np.zeros(values.size, dtype=bool)
    uncertain_mask = np.zeros(values.size, dtype=bool)
    contact_weight = np.ones(values.size, dtype=float)
    state_confidence = np.ones(values.size, dtype=float)
    events: list[dict] = []
    initial_end = min(values.size, int(np.ceil(config.initial_contact_s * fs)))
    pilot_activity = _pilot_slope_activity(baseline, fs, initial_end, config)
    anchor_samples = max(1, int(round(config.contact_anchor_window_s * fs)))
    residual_noise_limit = _residual_noise_limit(values, baseline, initial_end, config)
    pending_detaches: list[int] = []
    pending_had_attach = False

    def pair_rejection(detach: dict, attach: dict) -> str | None:
        """返回一对候选无法形成完整非接触证据的首要原因。"""
        detach_start, detach_end = _candidate_bounds(detach, values.size)
        attach_start, attach_end = _candidate_bounds(attach, values.size)
        confidence = min(float(detach["confidence"]), float(attach["confidence"]))
        if confidence < config.state_transition_confidence:
            return "low_transition_confidence"
        if attach["direction"] == detach["direction"]:
            return "same_direction_attach"
        if float(attach["duration_s"]) <= float(detach["duration_s"]):
            return "attach_not_slower"
        if (attach_start - detach_end) / float(fs) < config.state_min_dwell_s:
            return "insufficient_dwell"

        before_start = max(0, detach_start - anchor_samples)
        after_end = min(values.size, attach_end + 1 + anchor_samples)
        before_activity = pilot_activity[before_start:detach_start]
        after_activity = pilot_activity[attach_end + 1 : after_end]
        anchors_stable = (
            before_activity.size == anchor_samples
            and after_activity.size == anchor_samples
            and _activity_fraction(before_activity) <= config.max_dynamic_fraction
            and _activity_fraction(after_activity) <= config.max_dynamic_fraction
            and _anchor_residual_is_stable(
                values, baseline, before_start, detach_start, anchor_samples, residual_noise_limit
            )
            and _anchor_residual_is_stable(
                values, baseline, attach_end + 1, after_end, anchor_samples, residual_noise_limit
            )
        )
        if not anchors_stable:
            return "unstable_contact_anchor"
        if _activity_fraction(pilot_activity[detach_end + 1 : attach_start]) > config.max_dynamic_fraction:
            return "excessive_pilot_slope_activity"
        return None

    for index, candidate in enumerate(updated):
        start, end = _candidate_bounds(candidate, values.size)
        candidate["accepted"] = False
        if start < initial_end:
            candidate["rejection_reason"] = "initial_contact_forced"
            uncertain_mask[start : end + 1] = True
            state_confidence[start : end + 1] = 0.0
            continue

        candidate_type = candidate["candidate_type"]
        if candidate_type == "detach":
            if float(candidate["confidence"]) < config.state_transition_confidence:
                candidate["rejection_reason"] = "low_transition_confidence"
            else:
                candidate["rejection_reason"] = "tentative_detach"
            pending_detaches.append(index)
            continue

        if candidate_type != "attach":
            candidate["rejection_reason"] = "not_a_tentative_detach"
            uncertain_mask[start : end + 1] = True
            state_confidence[start : end + 1] = 0.0
            continue

        if not pending_detaches:
            candidate["rejection_reason"] = "attach_without_tentative_detach"
            uncertain_mask[start : end + 1] = True
            state_confidence[start : end + 1] = 0.0
            continue

        pending_had_attach = True
        chosen_index = None
        for detach_index in reversed(pending_detaches):
            detach = updated[detach_index]
            reason = pair_rejection(detach, candidate)
            detach["rejection_reason"] = reason or "tentative_detach"
            if reason is None:
                chosen_index = detach_index
                break

        if chosen_index is None:
            first_start, _ = _candidate_bounds(updated[pending_detaches[0]], values.size)
            candidate["rejection_reason"] = updated[pending_detaches[-1]]["rejection_reason"]
            uncertain_mask[first_start : end + 1] = True
            state_confidence[first_start : end + 1] = 0.0
            continue

        detach = updated[chosen_index]
        detach_start, detach_end = _candidate_bounds(detach, values.size)
        for detach_index in pending_detaches:
            if detach_index != chosen_index:
                updated[detach_index]["rejection_reason"] = "ignored_inside_confirmed_interval"
        detach["accepted"] = True
        detach["rejection_reason"] = None
        candidate["accepted"] = True
        candidate["rejection_reason"] = None
        uncertain_mask[detach_start : end + 1] = False
        transition_mask[detach_start : detach_end + 1] = True
        transition_mask[start : end + 1] = True
        noncontact_mask[detach_end + 1 : start] = True
        contact_mask[detach_start : end + 1] = False
        contact_weight[detach_start : detach_end + 1] = np.linspace(
            1.0, 0.0, detach_end - detach_start + 1
        )
        contact_weight[detach_end + 1 : start] = 0.0
        contact_weight[start : end + 1] = np.linspace(0.0, 1.0, end - start + 1)
        confidence = min(float(detach["confidence"]), float(candidate["confidence"]))
        state_confidence[detach_start : end + 1] = confidence
        events.extend(({**detach, "event_type": "detach"}, {**candidate, "event_type": "attach"}))
        pending_detaches.clear()
        pending_had_attach = False

    # 只有从未出现匹配 attach 的最后一个高置信 tentative detach 才能形成尾段非接触。
    if pending_detaches and not pending_had_attach:
        detach = updated[pending_detaches[-1]]
        detach_start, detach_end = _candidate_bounds(detach, values.size)
        before_start = max(0, detach_start - anchor_samples)
        before_activity = pilot_activity[before_start:detach_start]
        tail_activity = pilot_activity[detach_end + 1 :]
        terminal_ok = (
            float(detach["confidence"]) >= config.state_transition_confidence
            and (values.size - detach_end - 1) / float(fs) >= config.terminal_min_tail_s
            and _activity_fraction(before_activity) <= config.max_dynamic_fraction
            and _activity_fraction(tail_activity) <= config.max_dynamic_fraction
            and _anchor_residual_is_stable(
                values, baseline, before_start, detach_start, anchor_samples, residual_noise_limit
            )
        )
        if terminal_ok:
            detach["accepted"] = True
            detach["rejection_reason"] = None
            transition_mask[detach_start : detach_end + 1] = True
            noncontact_mask[detach_end + 1 :] = True
            contact_mask[detach_start:] = False
            contact_weight[detach_start : detach_end + 1] = np.linspace(
                1.0, 0.0, detach_end - detach_start + 1
            )
            contact_weight[detach_end + 1 :] = 0.0
            state_confidence[detach_start:] = float(detach["confidence"])
            events.append({**detach, "event_type": "detach"})
        else:
            detach["rejection_reason"] = "unpaired_or_unstable_terminal_detach"

    for detach_index in pending_detaches:
        detach = updated[detach_index]
        if not detach["accepted"]:
            start, end = _candidate_bounds(detach, values.size)
            if detach["rejection_reason"] == "tentative_detach":
                detach["rejection_reason"] = "unpaired_detach"
            uncertain_mask[start : end + 1] = True
            state_confidence[start : end + 1] = 0.0

    contact_mask[:initial_end] = True
    noncontact_mask[:initial_end] = False
    transition_mask[:initial_end] = False
    contact_weight[:initial_end] = 1.0
    return {
        "contact_mask": contact_mask,
        "noncontact_mask": noncontact_mask,
        "transition_mask": transition_mask,
        "uncertain_mask": uncertain_mask,
        "contact_weight": contact_weight,
        "state_confidence": state_confidence,
        "J_events": events,
        "candidates": updated,
    }


def reconstruct_contact_mode(
    clean: np.ndarray,
    state_result: dict,
    fs: float,
    config: ContactStateConfig,
) -> dict:
    """以局部稳健接触锚点填补已接受的内部非接触区间。"""
    values = _validated_signal(clean, fs, config)
    if not isinstance(state_result, dict):
        raise TypeError("state_result 必须是字典")
    output = values.copy()
    synthetic = np.zeros(values.size, dtype=bool)
    events = state_result.get("J_events", [])
    anchor_samples = max(1, int(round(config.contact_anchor_window_s * fs)))

    event_index = 0
    while event_index + 1 < len(events):
        detach = events[event_index]
        attach = events[event_index + 1]
        if detach.get("event_type") != "detach" or attach.get("event_type") != "attach":
            event_index += 1
            continue
        detach_start, _ = _candidate_bounds(detach, values.size)
        _, attach_end = _candidate_bounds(attach, values.size)
        before = values[max(0, detach_start - anchor_samples) : detach_start]
        after = values[attach_end + 1 : min(values.size, attach_end + 1 + anchor_samples)]
        if before.size and after.size:
            output[detach_start : attach_end + 1] = np.linspace(
                float(np.median(before)),
                float(np.median(after)),
                attach_end - detach_start + 1,
            )
            synthetic[detach_start : attach_end + 1] = True
        event_index += 2

    if events and events[-1].get("event_type") == "detach":
        detach_start, _ = _candidate_bounds(events[-1], values.size)
        before = values[max(0, detach_start - anchor_samples) : detach_start]
        if before.size:
            output[detach_start:] = float(np.median(before))
            synthetic[detach_start:] = True

    return {"Ucontact": output, "ucontact_synthetic_mask": synthetic}


def estimate_contact_bm(
    ucontact: np.ndarray,
    fs: float,
    config: ContactStateConfig,
) -> dict:
    """从接触模态重建信号估计最终中低频趋势 ``Bm_contact``。

    这里的 LOWESS 只处理已修复非接触缺口的 ``Ucontact``，因此不会再依赖
    易累积误差的 ``clean - J``。窗口决定保留的最短中尺度变化，稳健迭代则
    降低残余离群点对局部回归的影响。
    """
    values = _validated_signal(ucontact, fs, config)
    if values.size < 3:
        raise ValueError("估计接触趋势至少需要三个采样点")

    time = np.arange(values.size, dtype=float) / float(fs)
    duration_s = values.size / float(fs)
    fraction = min(
        1.0,
        max(3.0 / values.size, config.contact_lowess_window_s / duration_s),
    )
    bm_contact = np.asarray(
        lowess(
            values,
            time,
            frac=fraction,
            it=config.contact_lowess_robust_iterations,
            return_sorted=False,
        ),
        dtype=float,
    )
    return {
        "Bm_contact": bm_contact,
        "dBm_dt": np.gradient(bm_contact) * float(fs),
    }


def estimate_conditional_bslow(
    bm_contact: np.ndarray,
    state_result: dict,
    diagnostics: dict,
    fs: float,
    config: ContactStateConfig,
) -> dict:
    """仅在真实、稳定接触点更新超慢基线。

    非接触、转换、模糊判断和人工重建区均严格冻结。稳定接触点还需同时满足
    边沿、趋势斜率和相对上一时刻慢基线的偏移阈值，避免真实压力被慢基线
    逐渐吸收。
    """
    bm = _validated_signal(bm_contact, fs, config)
    contact = np.asarray(state_result["contact_mask"], dtype=bool)
    noncontact = np.asarray(state_result["noncontact_mask"], dtype=bool)
    transition = np.asarray(state_result["transition_mask"], dtype=bool)
    uncertain = np.asarray(state_result["uncertain_mask"], dtype=bool)
    synthetic = np.asarray(
        diagnostics.get("ucontact_synthetic_mask", np.zeros(bm.size, dtype=bool)),
        dtype=bool,
    )
    slope = np.asarray(diagnostics["dBm_dt"], dtype=float)
    edge_score = np.asarray(
        diagnostics.get("edge_score", diagnostics.get("scores", np.zeros(bm.size))),
        dtype=float,
    )
    arrays = (contact, noncontact, transition, uncertain, synthetic, slope, edge_score)
    if any(values.shape != bm.shape for values in arrays):
        raise ValueError("状态掩码和诊断数组必须与 Bm_contact 等长")

    alpha = config.bslow_update_rate * (
        1.0 - np.exp(-1.0 / (float(fs) * config.bslow_tau_s))
    )
    bslow = np.empty_like(bm)
    update_mask = np.zeros(bm.size, dtype=bool)
    bslow[0] = bm[0]

    real_stable_contact = contact & ~noncontact & ~transition & ~uncertain & ~synthetic
    for index in range(1, bm.size):
        stable = (
            real_stable_contact[index]
            and abs(edge_score[index]) < config.bslow_stable_edge_threshold
            and abs(slope[index]) < config.bslow_stable_slope_threshold
            and abs(bm[index] - bslow[index - 1])
            < config.bslow_stable_offset_threshold
        )
        if stable:
            bslow[index] = bslow[index - 1] + alpha * (bm[index] - bslow[index - 1])
            update_mask[index] = True
        else:
            bslow[index] = bslow[index - 1]

    return {"Bslow": bslow, "bslow_update_mask": update_mask}


def decompose_contact_state_udeco(
    signal: np.ndarray,
    fs: float,
    config: ContactStateConfig | None = None,
) -> dict:
    """执行单路 Udeco 的完整接触状态分解。

    每次调用都从原信号独立估计噪声、候选和状态；函数不保存跨通道状态，
    因而可用同一接口分别处理 Udeco2 与 Udeco3。
    """
    config = ContactStateConfig() if config is None else config
    raw = _validated_signal(signal, fs, config).copy()
    clean = hampel_filter(raw, fs, config)
    pilot = estimate_pilot_trend(clean, fs, config)
    candidates, candidate_diagnostics = detect_transition_candidates(
        clean, pilot["Bpilot"], fs, config
    )
    state = infer_contact_states(clean, pilot["Bpilot"], candidates, fs, config)
    reconstruction = reconstruct_contact_mode(clean, state, fs, config)
    medium = estimate_contact_bm(reconstruction["Ucontact"], fs, config)

    slow_diagnostics = {
        "dBm_dt": medium["dBm_dt"],
        "edge_score": candidate_diagnostics["scores"],
        "ucontact_synthetic_mask": reconstruction["ucontact_synthetic_mask"],
    }
    slow = estimate_conditional_bslow(
        medium["Bm_contact"], state, slow_diagnostics, fs, config
    )

    p_candidate = medium["Bm_contact"] - slow["Bslow"]
    p = p_candidate * state["contact_weight"]
    p[state["noncontact_mask"]] = 0.0
    p_valid_mask = (
        state["contact_mask"]
        & ~state["transition_mask"]
        & ~state["uncertain_mask"]
        & ~reconstruction["ucontact_synthetic_mask"]
    )

    # J 只描述当前非接触模态与边沿，不累计历史阶跃。短中值窗可避免把
    # 高频测量噪声全部塞进 J；剩余部分自然进入 residual。
    mode_active = state["noncontact_mask"] | state["transition_mask"]
    mode_signal = clean - slow["Bslow"] - p
    smooth_size = max(1, int(round(config.j_smooth_window_s * fs)) | 1)
    smooth_mode_signal = median_filter(mode_signal, size=smooth_size, mode="nearest")
    j = np.zeros_like(clean)
    j[mode_active] = smooth_mode_signal[mode_active]
    residual = clean - j - p - slow["Bslow"]

    return {
        "raw": raw,
        "clean": clean,
        **pilot,
        "transition_candidates": state["candidates"],
        "candidate_score": candidate_diagnostics["scores"],
        "J_events": state["J_events"],
        "contact_mask": state["contact_mask"],
        "noncontact_mask": state["noncontact_mask"],
        "transition_mask": state["transition_mask"],
        "uncertain_mask": state["uncertain_mask"],
        "contact_weight": state["contact_weight"],
        "state_confidence": state["state_confidence"],
        **reconstruction,
        **medium,
        **slow,
        "P": p,
        "P_valid_mask": p_valid_mask,
        "J": j,
        "mode_gap_raw": clean - reconstruction["Ucontact"],
        "residual": residual,
    }
