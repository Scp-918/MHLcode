"""基于边沿形态和级联状态机的 Udeco 单通道实验算法。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter
from statsmodels.nonparametric.smoothers_lowess import lowess


@dataclass(frozen=True)
class CascadeStateConfig:
    """集中保存秒制窗口和具有明确物理意义的形态阈值。"""

    median_filter_points: int = 4
    context_window_s: float = 10.0
    context_robust_iterations: int = 1
    edge_scales_s: tuple[float, ...] = (
        0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56,
    )
    edge_z_threshold: float = 4.0
    edge_merge_s: float = 0.08
    edge_level_window_min_s: float = 0.04
    edge_level_window_max_s: float = 0.20
    edge_guard_s: float = 0.02
    detach_direction: str = "up"
    detach_max_points: int = 4
    detach_min_snr: float = 5.0
    detach_min_relative_amplitude: float = 0.03
    detach_pre_level_fraction: float = -0.20
    attach_direction: str = "down"
    attach_min_points: int = 5
    attach_min_monotonicity: float = 0.65
    attach_min_snr: float = 4.0
    attach_min_relative_amplitude: float = 0.01
    edge_min_persistence: float = 0.60
    edge_asymmetry_ratio: float = 2.0
    mode_level_noise_ratio: float = 2.0
    mode_level_delta_fraction: float = 0.35
    state_min_confidence: float = 0.55
    state_refractory_s: float = 0.08
    uncertain_margin_s: float = 0.05
    contact_anchor_window_s: float = 1.0
    contact_lowess_window_s: float = 3.0
    contact_robust_iterations: int = 2
    bslow_tau_s: float = 60.0
    bslow_update_rate: float = 1.0
    bslow_stable_slope_threshold: float = 3e-4
    bslow_stable_offset_threshold: float = 1e-3
    pressure_slope_threshold: float = 3e-4
    pressure_offset_threshold: float = 1e-3
    j_smooth_window_s: float = 0.10


def _as_signal(values: np.ndarray) -> np.ndarray:
    """将输入转成有限的一维浮点数组。"""
    signal = np.asarray(values, dtype=float)
    if signal.ndim != 1 or signal.size < 3 or not np.isfinite(signal).all():
        raise ValueError("信号必须是至少三个有限采样点组成的一维数组")
    return signal


def _mad(values: np.ndarray) -> float:
    """返回未缩放的中位数绝对偏差。"""
    center = float(np.median(values))
    return float(np.median(np.abs(values - center)))


def median_clean(signal: np.ndarray, config: CascadeStateConfig) -> np.ndarray:
    """使用短中值窗去掉孤立尖峰，同时保留具有持续新平台的边沿。"""
    values = _as_signal(signal)
    size = max(1, int(config.median_filter_points))
    return median_filter(values, size=size, mode="nearest")


def estimate_context_trend(
    clean: np.ndarray,
    fs: float,
    config: CascadeStateConfig,
) -> dict:
    """估计只服务于边沿定位的低频上下文 ``Bcontext``。

    LOWESS 是局部一次线性回归。默认十秒窗口明显慢于接触边沿，也比最终
    压力趋势窗口更长；它不负责直接把压力从信号中删除。
    """
    values = _as_signal(clean)
    duration_s = values.size / float(fs)
    fraction = min(1.0, max(3.0 / values.size, config.context_window_s / duration_s))
    time = np.arange(values.size, dtype=float) / float(fs)
    context = np.asarray(
        lowess(
            values,
            time,
            frac=fraction,
            it=int(config.context_robust_iterations),
            return_sorted=False,
        ),
        dtype=float,
    )
    return {
        "Bcontext": context,
        "edge_signal": values - context,
        "context_slope": np.gradient(context) * float(fs),
    }


def _rolling_noise(values: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
    """估计局部噪声，并以全段最安静窗口的统计量提供稳定下限。"""
    window = max(5, int(round(0.20 * fs)) | 1)
    local_center = median_filter(values, size=window, mode="nearest")
    local_mad = median_filter(
        np.abs(values - local_center), size=window, mode="nearest"
    )
    positive = local_mad[local_mad > np.finfo(float).eps]
    quiet_noise = float(np.percentile(positive, 20)) if positive.size else 0.0
    # 完全平滑的合成信号也会留下 LOWESS 数值曲率；以动态范围的千分之一
    # 作为极小下限，可避免这些数值残差被放大成连续的伪边沿。
    reference = max(quiet_noise, float(np.ptp(values)) * 1e-3, np.finfo(float).eps)
    return local_mad, reference


def detect_multiscale_edges(
    clean: np.ndarray,
    bcontext: np.ndarray,
    fs: float,
    config: CascadeStateConfig,
) -> tuple[list[dict], dict]:
    """用多尺度稳健 Haar 差定位边沿，不在此阶段决定接触状态。"""
    values = _as_signal(clean)
    context = _as_signal(bcontext)
    if values.shape != context.shape:
        raise ValueError("clean 与 Bcontext 必须等长")

    edge_signal = values - context
    reference_level = max(float(np.median(np.abs(values))), np.finfo(float).eps)
    local_mad, reference_noise = _rolling_noise(edge_signal, fs)
    best_score = np.zeros(values.size)
    best_scale = np.zeros(values.size)
    first_score = np.zeros(values.size)
    first_scale = np.zeros(values.size)

    # 每个中心比较左、右等长窗口的中位数，相当于对噪声更稳健的 Haar 核。
    for scale_s in config.edge_scales_s:
        width = max(2, int(round(scale_s * fs)))
        if 2 * width >= values.size:
            continue
        rolling_median = np.median(sliding_window_view(edge_signal, width), axis=1)
        centers = np.arange(width, values.size - width + 1)
        change = rolling_median[centers] - rolling_median[centers - width]
        noise = np.maximum(local_mad[centers], reference_noise)
        score = change / np.maximum(1.4826 * noise, np.finfo(float).eps)

        stronger = np.abs(score) > np.abs(best_score[centers])
        selected = centers[stronger]
        best_score[selected] = score[stronger]
        best_scale[selected] = scale_s

        first = (first_scale[centers] == 0.0) & (
            np.abs(score) >= config.edge_z_threshold
        )
        selected = centers[first]
        first_score[selected] = score[first]
        first_scale[selected] = scale_s

    active = np.flatnonzero(first_scale > 0.0)
    if active.size == 0:
        return [], {
            "edge_signal": edge_signal,
            "edge_score": best_score,
            "reference_noise": reference_noise,
        }

    merge_gap = max(1, int(round(config.edge_merge_s * fs)))
    groups: list[list[int]] = [[int(active[0])]]
    for index in active[1:]:
        previous = groups[-1][-1]
        same_direction = np.sign(first_score[index]) == np.sign(first_score[previous])
        if index - previous <= merge_gap and same_direction:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])

    candidates = []
    for group in groups:
        indices = np.asarray(group)
        smallest_scale = float(np.min(first_scale[indices]))
        fine = indices[first_scale[indices] == smallest_scale]
        center = int(fine[np.argmax(np.abs(first_score[fine]))])
        candidates.append(
            {
                "center_index": center,
                "coarse_start_index": int(indices[0]),
                "coarse_end_index": int(indices[-1]),
                "best_scale_s": smallest_scale,
                "edge_score": float(first_score[center]),
                "noise_scale": float(max(local_mad[center], reference_noise)),
                "reference_level": reference_level,
            }
        )

    return candidates, {
        "edge_signal": edge_signal,
        "edge_score": best_score,
        "reference_noise": reference_noise,
    }


def measure_edge_morphology(
    edge_signal: np.ndarray,
    candidate: dict,
    fs: float,
    config: CascadeStateConfig,
    reference_noise: float,
) -> dict:
    """以固定短窗 L/R 和一次 10%–90% 穿越细化边沿形态。"""
    values = _as_signal(edge_signal)
    center = int(candidate["center_index"])
    scale_points = max(2, int(round(float(candidate["best_scale_s"]) * fs)))
    minimum = max(2, int(round(config.edge_level_window_min_s * fs)))
    maximum = max(minimum, int(round(config.edge_level_window_max_s * fs)))
    level_points = int(np.clip(2 * scale_points, minimum, maximum))
    guard = max(1, int(round(config.edge_guard_s * fs)))
    # 至少向两侧跨过一个完整的最大平台窗。慢 attach 的 Haar 峰常靠近
    # 变化起点；若只查看半个窗口，会把尚未完成的恢复误量成快速边沿。
    half_span = max(maximum, 2 * scale_points)

    pre_end = max(0, center - half_span)
    post_start = min(values.size, center + half_span)
    pre = values[max(0, pre_end - level_points) : pre_end]
    post = values[post_start : min(values.size, post_start + level_points)]
    if pre.size < 2 or post.size < 2:
        return {**candidate, "candidate_type": "ambiguous", "rejection_reason": "edge_near_boundary"}

    left = float(np.median(pre))
    right = float(np.median(post))
    delta = right - left
    if abs(delta) <= np.finfo(float).eps:
        return {**candidate, "candidate_type": "ambiguous", "rejection_reason": "no_level_change"}

    search_start = max(0, center - half_span)
    search_end = min(values.size - 1, center + half_span)
    progress = (values[search_start : search_end + 1] - left) / delta
    local_center = center - search_start
    before = np.flatnonzero(progress[: local_center + 1] <= 0.10)
    after = np.flatnonzero(progress[local_center:] >= 0.90)
    start = search_start + int(before[-1]) if before.size else max(0, center - 1)
    end = center + int(after[0]) if after.size else min(values.size - 1, center + 1)
    if end <= start:
        start, end = max(0, center - 1), min(values.size - 1, center + 1)

    # 最终 L/R 窗位于细化边沿之外；只更新平台量，不再循环移动边界。
    final_pre_end = max(0, start - guard)
    final_post_start = min(values.size, end + guard + 1)
    final_pre = values[max(0, final_pre_end - level_points) : final_pre_end]
    final_post = values[final_post_start : min(values.size, final_post_start + level_points)]
    if final_pre.size >= 2:
        left = float(np.median(final_pre))
    if final_post.size >= 2:
        right = float(np.median(final_post))
    delta = right - left
    relative_amplitude = abs(float(delta)) / max(
        float(candidate.get("reference_level", 1.0)), np.finfo(float).eps
    )

    transition = values[start : end + 1]
    signed_steps = np.diff(transition) * (1.0 if delta >= 0 else -1.0)
    morphology_noise = max(
        float(reference_noise),
        float(candidate.get("noise_scale", reference_noise)),
        np.finfo(float).eps,
    )
    tolerance = 1.4826 * morphology_noise
    monotonicity = float(np.mean(signed_steps >= -tolerance)) if signed_steps.size else 1.0
    follow = values[end + 1 : min(values.size, end + 1 + level_points)]
    if follow.size:
        persistence = float(np.mean((follow - left) / delta >= 0.5))
    else:
        persistence = 0.0

    return {
        **candidate,
        "start_index": start,
        "end_index": end,
        "start_s": start / float(fs),
        "end_s": end / float(fs),
        "edge_width_points": end - start,
        "edge_width_s": (end - start) / float(fs),
        "level_left": left,
        "level_right": right,
        "level_delta": float(delta),
        "level_snr": abs(float(delta)) / (1.4826 * morphology_noise),
        "morphology_noise": morphology_noise,
        "relative_amplitude": relative_amplitude,
        "direction": "up" if delta > 0 else "down",
        "monotonicity": monotonicity,
        "persistence": persistence,
        "candidate_type": "unclassified",
        "accepted": False,
        "rejection_reason": "awaiting_classification",
    }


def classify_edge_candidates(
    candidates: list[dict],
    config: CascadeStateConfig,
) -> list[dict]:
    """按方向、点数和持续性将候选分成 detach/attach/pressure-like。"""
    classified = []
    for source in candidates:
        item = dict(source)
        if item.get("candidate_type") == "ambiguous":
            classified.append(item)
            continue
        direction = item["direction"]
        width = int(item["edge_width_points"])
        snr = float(item["level_snr"])
        persistence = float(item["persistence"])
        relative_amplitude = float(item["relative_amplitude"])
        positive_limit = config.mode_level_noise_ratio * float(item["morphology_noise"])
        detach_destination = (
            float(item["level_right"]) >= positive_limit
            and float(item["level_right"]) >= config.mode_level_delta_fraction * abs(float(item["level_delta"]))
        )
        detach_origin = float(item["level_left"]) >= (
            config.detach_pre_level_fraction * abs(float(item["level_delta"]))
        )
        attach_source = (
            float(item["level_left"]) >= positive_limit
            and float(item["level_left"]) >= config.mode_level_delta_fraction * abs(float(item["level_delta"]))
        )
        if (
            direction == config.detach_direction
            and width <= config.detach_max_points
            and snr >= config.detach_min_snr
            and relative_amplitude >= config.detach_min_relative_amplitude
            and persistence >= config.edge_min_persistence
            and detach_destination
            and detach_origin
        ):
            kind = "detach"
        elif (
            direction == config.attach_direction
            and width >= config.attach_min_points
            and snr >= config.attach_min_snr
            and relative_amplitude >= config.attach_min_relative_amplitude
            and float(item["monotonicity"]) >= config.attach_min_monotonicity
            and attach_source
        ):
            kind = "attach"
        elif width > config.detach_max_points:
            kind = "pressure_like"
        else:
            kind = "ambiguous"

        edge_ratio = abs(float(item["edge_score"])) / config.edge_z_threshold
        snr_limit = config.detach_min_snr if kind == "detach" else config.attach_min_snr
        confidence = np.clip(
            (min(edge_ratio, 2.0) / 2.0 + min(snr / snr_limit, 2.0) / 2.0)
            * 0.5
            * max(float(item["monotonicity"]), persistence),
            0.0,
            1.0,
        )
        item.update(
            candidate_type=kind,
            confidence=float(confidence),
            rejection_reason="awaiting_state_inference",
        )
        classified.append(item)
    return classified


def infer_contact_mode(
    candidates: list[dict],
    sample_count: int,
    fs: float,
    config: CascadeStateConfig,
) -> dict:
    """用独立边沿驱动 Contact/Non-contact 状态，而非等待完整事件对。

    第一个可靠边沿同时确定记录开头的状态。detach 在 Contact 中立即生效；
    attach 在 Non-contact 中生效，并在存在前序 detach 时检查恢复边沿确实更慢。
    """
    updated = [dict(item) for item in sorted(candidates, key=lambda x: x.get("center_index", 0))]
    reliable = [
        item for item in updated
        if item.get("candidate_type") in {"detach", "attach"}
        and float(item.get("confidence", 0.0)) >= config.state_min_confidence
    ]
    current_contact = not reliable or reliable[0]["candidate_type"] == "detach"
    initial_contact = current_contact
    accepted: list[dict] = []
    last_detach: dict | None = None
    last_event_end = -10**9
    refractory = max(1, int(round(config.state_refractory_s * fs)))

    for item in updated:
        item["accepted"] = False
        kind = item.get("candidate_type")
        confidence = float(item.get("confidence", 0.0))
        if kind not in {"detach", "attach"}:
            item["rejection_reason"] = "pressure_or_ambiguous"
            continue
        if confidence < config.state_min_confidence:
            item["rejection_reason"] = "low_confidence"
            continue
        if int(item["start_index"]) - last_event_end < refractory:
            item["rejection_reason"] = "inside_refractory_period"
            continue

        if kind == "detach" and current_contact:
            item["accepted"] = True
            item["rejection_reason"] = None
            item["event_type"] = "detach"
            accepted.append(item)
            current_contact = False
            last_detach = item
            last_event_end = int(item["end_index"])
        elif kind == "attach" and not current_contact:
            if last_detach is not None and int(item["edge_width_points"]) < int(
                np.ceil(config.edge_asymmetry_ratio * int(last_detach["edge_width_points"]))
            ):
                item["rejection_reason"] = "attach_not_slower"
                continue
            item["accepted"] = True
            item["rejection_reason"] = None
            item["event_type"] = "attach"
            accepted.append(item)
            current_contact = True
            last_detach = None
            last_event_end = int(item["end_index"])
        else:
            item["rejection_reason"] = "edge_incompatible_with_current_state"

    contact = np.zeros(sample_count, dtype=bool)
    noncontact = np.zeros(sample_count, dtype=bool)
    transition = np.zeros(sample_count, dtype=bool)
    uncertain = np.zeros(sample_count, dtype=bool)
    weight = np.zeros(sample_count, dtype=float)
    confidence = np.ones(sample_count, dtype=float)
    cursor = 0
    current_contact = initial_contact

    for event in accepted:
        start = max(cursor, int(event["start_index"]))
        end = min(sample_count - 1, int(event["end_index"]))
        if current_contact:
            contact[cursor:start] = True
            weight[cursor:start] = 1.0
        else:
            noncontact[cursor:start] = True
        transition[start : end + 1] = True
        if event["event_type"] == "detach":
            weight[start : end + 1] = np.linspace(1.0, 0.0, end - start + 1)
            current_contact = False
        else:
            weight[start : end + 1] = np.linspace(0.0, 1.0, end - start + 1)
            current_contact = True
        confidence[start : end + 1] = float(event["confidence"])
        cursor = end + 1

    if current_contact:
        contact[cursor:] = True
        weight[cursor:] = 1.0
    else:
        noncontact[cursor:] = True

    margin = max(0, int(round(config.uncertain_margin_s * fs)))
    for item in updated:
        if item["accepted"] or item.get("candidate_type") == "pressure_like":
            continue
        start = max(0, int(item.get("start_index", item.get("center_index", 0))) - margin)
        end = min(sample_count, int(item.get("end_index", item.get("center_index", 0))) + margin + 1)
        uncertain[start:end] = True
        confidence[start:end] = 0.0

    return {
        "contact_mask": contact,
        "noncontact_mask": noncontact,
        "transition_mask": transition,
        "uncertain_mask": uncertain,
        "contact_weight": weight,
        "state_confidence": confidence,
        "J_events": accepted,
        "edge_candidates": updated,
    }


def reconstruct_contact_signal(
    clean: np.ndarray,
    state: dict,
    fs: float,
    config: CascadeStateConfig,
) -> dict:
    """用最近的真实 Contact 锚点重建 Non-contact/Transition 区域。"""
    values = _as_signal(clean)
    active = np.asarray(state["noncontact_mask"]) | np.asarray(state["transition_mask"])
    output = values.copy()
    synthetic = np.zeros(values.size, dtype=bool)
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return {"Ucontact": output, "ucontact_synthetic_mask": synthetic}

    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    anchor = max(1, int(round(config.contact_anchor_window_s * fs)))
    contact = np.asarray(state["contact_mask"], dtype=bool)
    for group in groups:
        start, end = int(group[0]), int(group[-1])
        left_indices = np.flatnonzero(contact[max(0, start - anchor) : start]) + max(0, start - anchor)
        right_indices = np.flatnonzero(contact[end + 1 : min(values.size, end + 1 + anchor)]) + end + 1
        if left_indices.size and right_indices.size:
            left = float(np.median(values[left_indices]))
            right = float(np.median(values[right_indices]))
            output[start : end + 1] = np.linspace(left, right, end - start + 1)
        elif left_indices.size:
            output[start : end + 1] = float(np.median(values[left_indices]))
        elif right_indices.size:
            output[start : end + 1] = float(np.median(values[right_indices]))
        else:
            continue
        synthetic[start : end + 1] = True
    return {"Ucontact": output, "ucontact_synthetic_mask": synthetic}


def estimate_contact_trends(
    ucontact: np.ndarray,
    state: dict,
    synthetic_mask: np.ndarray,
    fs: float,
    config: CascadeStateConfig,
) -> dict:
    """从接触模态信号估计 Bm，并仅在真实稳定 Contact 中更新 Bslow。"""
    values = _as_signal(ucontact)
    time = np.arange(values.size, dtype=float) / float(fs)
    duration_s = values.size / float(fs)
    fraction = min(
        1.0,
        max(3.0 / values.size, config.contact_lowess_window_s / duration_s),
    )
    bm = np.asarray(
        lowess(
            values,
            time,
            frac=fraction,
            it=int(config.contact_robust_iterations),
            return_sorted=False,
        ),
        dtype=float,
    )
    slope = np.gradient(bm) * float(fs)
    real_contact = (
        np.asarray(state["contact_mask"], dtype=bool)
        & ~np.asarray(state["uncertain_mask"], dtype=bool)
        & ~np.asarray(synthetic_mask, dtype=bool)
    )
    real_indices = np.flatnonzero(real_contact)
    first = int(real_indices[0]) if real_indices.size else 0
    bslow = np.full(values.size, bm[first], dtype=float)
    update = np.zeros(values.size, dtype=bool)
    alpha = config.bslow_update_rate * (
        1.0 - np.exp(-1.0 / (float(fs) * config.bslow_tau_s))
    )
    for index in range(max(1, first + 1), values.size):
        stable = (
            real_contact[index]
            and abs(slope[index]) < config.bslow_stable_slope_threshold
            and abs(bm[index] - bslow[index - 1]) < config.bslow_stable_offset_threshold
        )
        if stable:
            bslow[index] = bslow[index - 1] + alpha * (bm[index] - bslow[index - 1])
            update[index] = True
        else:
            bslow[index] = bslow[index - 1]
    return {
        "Bm_contact": bm,
        "dBm_dt": slope,
        "Bslow": bslow,
        "bslow_update_mask": update,
    }


def classify_contact_substates(
    p: np.ndarray,
    dBm_dt: np.ndarray,
    state: dict,
    config: CascadeStateConfig,
) -> np.ndarray:
    """仅在可靠 Contact 中按 P 与斜率定性标记压力阶段。"""
    pressure = np.asarray(p)
    slope = np.asarray(dBm_dt)
    valid = np.asarray(state["contact_mask"]) & ~np.asarray(state["uncertain_mask"])
    labels = np.full(pressure.size, "invalid_noncontact", dtype=object)
    quiet = np.abs(slope) <= config.pressure_slope_threshold
    near = np.abs(pressure) <= config.pressure_offset_threshold
    labels[valid & quiet & near] = "baseline"
    labels[valid & quiet & ~near] = "pressure_hold"
    moving = valid & ~quiet
    labels[moving & (pressure * slope >= 0.0)] = "loading"
    labels[moving & (pressure * slope < 0.0)] = "relaxing"
    return labels


def decompose_cascade_udeco(
    signal: np.ndarray,
    fs: float,
    config: CascadeStateConfig | None = None,
) -> dict:
    """执行形态边沿、接触状态和 Contact 内压力状态的完整级联分解。"""
    config = CascadeStateConfig() if config is None else config
    raw = _as_signal(signal).copy()
    clean = median_clean(raw, config)
    context = estimate_context_trend(clean, fs, config)
    coarse, edge_diagnostics = detect_multiscale_edges(
        clean, context["Bcontext"], fs, config
    )
    measured = [
        measure_edge_morphology(
            edge_diagnostics["edge_signal"],
            candidate,
            fs,
            config,
            edge_diagnostics["reference_noise"],
        )
        for candidate in coarse
    ]
    candidates = classify_edge_candidates(measured, config)
    state = infer_contact_mode(candidates, raw.size, fs, config)
    reconstruction = reconstruct_contact_signal(clean, state, fs, config)
    trends = estimate_contact_trends(
        reconstruction["Ucontact"],
        state,
        reconstruction["ucontact_synthetic_mask"],
        fs,
        config,
    )

    p_candidate = trends["Bm_contact"] - trends["Bslow"]
    p = p_candidate * state["contact_weight"]
    p[state["noncontact_mask"]] = 0.0
    substates = classify_contact_substates(p, trends["dBm_dt"], state, config)

    mode_active = state["noncontact_mask"] | state["transition_mask"]
    mode_signal = clean - trends["Bslow"] - p
    smooth_points = max(1, int(round(config.j_smooth_window_s * fs)) | 1)
    smooth_mode = median_filter(mode_signal, size=smooth_points, mode="nearest")
    j = np.zeros_like(clean)
    j[mode_active] = smooth_mode[mode_active]
    residual = clean - j - p - trends["Bslow"]

    return {
        "raw": raw,
        "clean": clean,
        **context,
        "edge_score": edge_diagnostics["edge_score"],
        "reference_noise": edge_diagnostics["reference_noise"],
        "edge_candidates": state["edge_candidates"],
        "J_events": state["J_events"],
        "contact_mask": state["contact_mask"],
        "noncontact_mask": state["noncontact_mask"],
        "transition_mask": state["transition_mask"],
        "uncertain_mask": state["uncertain_mask"],
        "contact_weight": state["contact_weight"],
        "state_confidence": state["state_confidence"],
        **reconstruction,
        **trends,
        "P": p,
        "contact_substate": substates,
        "J": j,
        "residual": residual,
    }
