"""基于局部信号证据的 Udeco 单通道边沿检测与分解。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter
from statsmodels.nonparametric.smoothers_lowess import lowess


@dataclass(frozen=True)
class CascadeStateConfig:
    """独立边沿检测及后续分解实际使用的参数。"""

    median_filter_points: int = 3
    context_window_s: float = 10.0
    context_robust_iterations: int = 1
    edge_scales_s: tuple[float, ...] = (
        0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56,
    )
    edge_z_threshold: float = 3.0
    edge_single_scale_z_threshold: float = 4.0
    edge_derivative_z_threshold: float = 4.0
    edge_min_scale_support: int = 2
    edge_merge_s: float = 0.08
    edge_fit_min_window_s: float = 0.25
    edge_fit_max_window_s: float = 1.50
    edge_min_snr: float = 3.0
    edge_min_fit_improvement: float = 0.15
    edge_min_plateau_contrast: float = 0.50
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
    local_mad, reference_noise = _rolling_noise(edge_signal, fs)
    best_score = np.zeros(values.size)
    scale_scores: list[tuple[float, np.ndarray]] = []

    # 每个中心比较左、右等长窗口的中位数，相当于对噪声更稳健的 Haar 核。
    for scale_s in config.edge_scales_s:
        width = max(2, int(round(scale_s * fs)))
        if 2 * width >= values.size:
            continue
        rolling_median = np.median(sliding_window_view(edge_signal, width), axis=1)
        centers = np.arange(width, values.size - width + 1)
        change = rolling_median[centers] - rolling_median[centers - width]
        # 边沿中心的滚动 MAD 会把真实跃迁当噪声；候选阶段取两侧安静
        # 平台的较小 MAD，仍由全段噪声下限防止分母接近零。
        noise = np.maximum(
            np.minimum(local_mad[centers - width], local_mad[centers + width - 1]),
            reference_noise,
        )
        score = change / np.maximum(1.4826 * noise, np.finfo(float).eps)
        score_row = np.zeros(values.size)
        score_row[centers] = score
        scale_scores.append((scale_s, score_row))

        stronger = np.abs(score) > np.abs(best_score[centers])
        selected = centers[stronger]
        best_score[selected] = score[stronger]

    active = np.flatnonzero(np.abs(best_score) >= config.edge_z_threshold)
    merge_gap = max(1, int(round(config.edge_merge_s * fs)))
    groups: list[list[int]] = [[int(active[0])]] if active.size else []
    for index in active[1:]:
        previous = groups[-1][-1]
        same_direction = np.sign(best_score[index]) == np.sign(best_score[previous])
        if index - previous <= merge_gap and same_direction:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])

    candidates = []
    for group in groups:
        indices = np.asarray(group)
        direction = np.sign(best_score[indices[0]])
        supported = [
            i for i, (_, row) in enumerate(scale_scores)
            if np.any(direction * row[indices] >= config.edge_z_threshold)
        ]
        consecutive = longest_run = 1
        for left, right in zip(supported, supported[1:]):
            consecutive = consecutive + 1 if right - left == 1 else 1
            longest_run = max(longest_run, consecutive)
        has_neighboring_scales = longest_run >= max(2, config.edge_min_scale_support)
        peak_score = max(
            float(np.max(direction * scale_scores[i][1][indices])) for i in supported
        )
        if not (has_neighboring_scales or peak_score >= config.edge_single_scale_z_threshold):
            continue

        # 最细的有效尺度定位最快边沿；其它尺度只提供一致性证据。
        smallest_index = supported[0]
        smallest_scale, fine_score = scale_scores[smallest_index]
        center = int(indices[np.argmax(direction * fine_score[indices])])
        width = max(2, int(round(smallest_scale * fs)))
        flank_noise = max(
            float(min(local_mad[center - width], local_mad[center + width - 1])),
            reference_noise,
        )
        candidates.append(
            {
                "center_index": center,
                "coarse_start_index": int(indices[0]),
                "coarse_end_index": int(indices[-1]),
                "best_scale_s": smallest_scale,
                "edge_score": float(direction * peak_score),
                "scale_support": len(supported),
                "support_scales_s": tuple(float(scale_scores[i][0]) for i in supported),
                "noise_scale": flank_noise,
                "detection_channels": ("haar",),
            }
        )

    # 一阶差分补足 Haar 窗口可能抹平的 1–3 点阶跃。MAD 在局部差分上
    # 估计，参考噪声作下限；连续同号响应构成一个跃迁，不在这里指定事件类型。
    difference = np.diff(edge_signal)
    noise_window = max(5, int(round(0.20 * fs)) | 1)
    difference_center = median_filter(difference, size=noise_window, mode="nearest")
    difference_mad = median_filter(
        np.abs(difference - difference_center), size=noise_window, mode="nearest"
    )
    difference_noise = np.maximum(difference_mad, 0.5 * reference_noise)
    derivative_score = np.zeros(values.size)
    derivative_score[1:] = difference / (1.4826 * difference_noise)
    derivative_active = np.flatnonzero(
        np.abs(derivative_score) >= config.edge_derivative_z_threshold
    )
    derivative_groups: list[list[int]] = []
    for index in derivative_active:
        if (
            derivative_groups
            and index == derivative_groups[-1][-1] + 1
            and np.sign(derivative_score[index]) == np.sign(derivative_score[derivative_groups[-1][-1]])
        ):
            derivative_groups[-1].append(int(index))
        else:
            derivative_groups.append([int(index)])

    for group in derivative_groups:
        indices = np.asarray(group)
        center = int(indices[np.argmax(np.abs(derivative_score[indices]))])
        direction = float(np.sign(derivative_score[center]))
        # 只将跨通道、同方向且中心几乎重合的响应视作同一跃迁。
        # 两个相邻快速阶跃即使小于旧 edge_merge_s，也不能被合并。
        matching = [
            item for item in candidates
            if np.sign(item["edge_score"]) == direction
            and "derivative" not in item["detection_channels"]
            and int(item["coarse_start_index"]) - 2 <= center <= int(item["coarse_end_index"]) + 2
        ]
        if matching:
            item = min(matching, key=lambda row: abs(int(row["center_index"]) - center))
            item["center_index"] = center
            item["detection_channels"] = ("haar", "derivative")
            item["derivative_score"] = float(derivative_score[center])
            item["coarse_start_index"] = min(item["coarse_start_index"], int(indices[0]))
            item["coarse_end_index"] = max(item["coarse_end_index"], int(indices[-1]))
        else:
            candidates.append({
                "center_index": center,
                "coarse_start_index": int(indices[0]),
                "coarse_end_index": int(indices[-1]),
                "best_scale_s": 1.0 / float(fs),
                "edge_score": float(derivative_score[center]),
                "derivative_score": float(derivative_score[center]),
                "scale_support": 0,
                "support_scales_s": (),
                "noise_scale": float(difference_noise[center - 1]),
                "detection_channels": ("derivative",),
            })

    candidates.sort(key=lambda item: item["center_index"])
    return candidates, {
        "edge_signal": edge_signal,
        "edge_score": best_score,
        "derivative_score": derivative_score,
        "reference_noise": reference_noise,
    }


def _robust_linear_fit(design: np.ndarray, values: np.ndarray, noise: float) -> tuple[np.ndarray, np.ndarray, float]:
    """用少量 IRLS 迭代拟合局部分段模型，并返回 Huber 损失。"""
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    for _ in range(3):
        residual = values - design @ coefficients
        scale = max(noise, 1.4826 * _mad(residual), np.finfo(float).eps)
        cutoff = 1.345 * scale
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), np.finfo(float).eps))
        weighted = np.sqrt(weights)
        coefficients = np.linalg.lstsq(design * weighted[:, None], values * weighted, rcond=None)[0]
    residual = values - design @ coefficients
    cutoff = 1.345 * max(noise, 1.4826 * _mad(residual), np.finfo(float).eps)
    magnitude = np.abs(residual)
    loss = float(np.mean(np.where(
        magnitude <= cutoff, 0.5 * residual**2,
        cutoff * (magnitude - 0.5 * cutoff),
    )))
    return coefficients, residual, loss


def measure_edge_morphology(
    clean: np.ndarray,
    candidate: dict,
    fs: float,
    config: CascadeStateConfig,
    reference_noise: float,
    window_bounds: tuple[int, int] | None = None,
) -> dict:
    """在 clean 上拟合连续的左平台、斜坡、右平台，定位 0%/100% 端点。"""
    values = _as_signal(clean)
    center = int(candidate["center_index"])
    if not 0 <= center < values.size:
        raise ValueError("candidate center_index 超出信号范围")
    scale_s = float(candidate["best_scale_s"])
    radius_s = min(config.edge_fit_max_window_s, max(config.edge_fit_min_window_s, 3.0 * scale_s))
    radius = max(3, int(round(radius_s * fs)))
    lower, upper = (0, values.size - 1) if window_bounds is None else window_bounds
    lo = max(0, center - radius, int(lower))
    hi = min(values.size - 1, center + radius, int(upper))
    if hi - lo < 4 or center <= lo or center >= hi:
        return {
            **candidate, "candidate_type": "ambiguous", "confidence": 0.0,
            "accepted": False, "rejection_reason": "insufficient_local_window",
        }

    x = np.arange(lo, hi + 1, dtype=float)
    local = values[lo : hi + 1]
    noise = max(1.4826 * float(reference_noise), np.finfo(float).eps)
    null_design = np.column_stack((np.ones(x.size), x - center))
    _, _, null_loss = _robust_linear_fit(null_design, local, noise)

    # 两侧至少各留一个平台采样点；粗网格加局部细化将搜索量限制为数百次。
    start_min, start_max = lo + 1, min(hi - 2, center + 2)
    end_min, end_max = max(lo + 2, center - 2), hi - 1
    stride = max(1, (hi - lo) // 36)
    starts = sorted(set(range(start_min, start_max + 1, stride)) | {start_max})
    ends = sorted(set(range(end_min, end_max + 1, stride)) | {end_max})
    best: tuple[float, int, int, np.ndarray, np.ndarray, float] | None = None

    def consider(start: int, end: int) -> None:
        nonlocal best
        if end <= start or not start - 2 <= center <= end + 2:
            return
        ramp = np.clip((x - start) / (end - start), 0.0, 1.0)
        design = np.column_stack((
            np.ones(x.size), np.minimum(x - start, 0.0),
            ramp, np.maximum(x - end, 0.0),
        ))
        coefficients, residual, loss = _robust_linear_fit(design, local, noise)
        span = hi - lo
        slope_penalty = 0.01 * (
            (coefficients[1] * span) ** 2 + (coefficients[3] * span) ** 2
        )
        score = loss + slope_penalty
        if best is None or score < best[0]:
            best = (score, start, end, coefficients, residual, loss)

    for start in starts:
        for end in ends:
            consider(start, end)
    if best is not None and stride > 1:
        coarse_start, coarse_end = best[1:3]
        for start in range(max(start_min, coarse_start - stride), min(start_max, coarse_start + stride) + 1):
            for end in range(max(end_min, coarse_end - stride), min(end_max, coarse_end + stride) + 1):
                consider(start, end)
    if best is None:
        return {
            **candidate, "candidate_type": "ambiguous", "confidence": 0.0,
            "accepted": False, "rejection_reason": "no_segmented_fit",
        }

    _, start, end, coefficients, residual, fit_loss = best
    left_level = float(coefficients[0])
    right_level = float(coefficients[0] + coefficients[2])
    delta = right_level - left_level
    left_residual = residual[: start - lo + 1]
    right_residual = residual[end - lo :]
    left_mad = _mad(left_residual)
    right_mad = _mad(right_residual)
    noise_left = max(1.4826 * left_mad, noise)
    noise_right = max(1.4826 * right_mad, noise)
    edge_snr = abs(delta) / np.hypot(noise_left, noise_right)
    improvement = float(np.clip(1.0 - fit_loss / max(null_loss, np.finfo(float).eps), 0.0, 1.0))
    ramp_slope = delta / (end - start)
    plateau_contrast = float(np.clip(
        1.0 - max(abs(coefficients[1]), abs(coefficients[3]))
        / max(abs(ramp_slope), np.finfo(float).eps),
        0.0, 1.0,
    ))
    signed_steps = np.diff(values[start : end + 1]) * np.sign(delta)
    monotonicity = float(np.mean(signed_steps >= -max(noise_left, noise_right)))
    return {
        **candidate,
        "start_index": start, "end_index": end,
        "start_s": start / float(fs), "end_s": end / float(fs),
        "edge_width_points": end - start, "edge_width_s": (end - start) / float(fs),
        "left_level": left_level, "right_level": right_level, "delta": delta,
        "level_left": left_level, "level_right": right_level, "level_delta": delta,
        "left_mad": left_mad, "right_mad": right_mad,
        "noise_left": noise_left, "noise_right": noise_right,
        "edge_snr": float(edge_snr), "level_snr": float(edge_snr),
        "fit_residual": float(np.sqrt(np.mean(residual**2))),
        "fit_improvement": improvement,
        "left_slope": float(coefficients[1]), "right_slope": float(coefficients[3]),
        "ramp_slope": float(ramp_slope), "plateau_contrast": plateau_contrast,
        "monotonicity": monotonicity,
        "direction": "up" if delta > 0 else "down",
        "fit_window_start_index": lo, "fit_window_end_index": hi,
        "candidate_type": "unclassified", "accepted": False,
        "rejection_reason": "awaiting_classification",
    }


def classify_edge_candidates(
    candidates: list[dict],
    config: CascadeStateConfig,
) -> list[dict]:
    """只按拟合平台差的符号命名边沿，以局部证据决定是否可信。"""
    classified = []
    for source in candidates:
        item = dict(source)
        if item.get("candidate_type") == "ambiguous":
            item["classification_reasons"] = (item.get("rejection_reason", "ambiguous"),)
            item.setdefault("confidence", 0.0)
            classified.append(item)
            continue
        delta = float(item["delta"])
        snr = float(item["edge_snr"])
        improvement = float(item["fit_improvement"])
        plateau_contrast = float(item.get("plateau_contrast", 1.0))
        monotonicity = float(item["monotonicity"])
        confidence = float(np.clip(
            0.5 * min(1.0, snr / 6.0)
            + 0.4 * min(1.0, improvement / 0.5)
            + 0.1 * monotonicity,
            0.0, 1.0,
        ))
        reasons = tuple(reason for reason, weak in (
            ("low_edge_snr", snr < config.edge_min_snr),
            ("weak_fit_improvement", improvement < config.edge_min_fit_improvement),
            ("no_distinct_plateaus", plateau_contrast < config.edge_min_plateau_contrast),
            ("no_level_change", abs(delta) <= np.finfo(float).eps),
        ) if weak)
        item.update(
            candidate_type="ambiguous" if reasons else ("detach" if delta > 0 else "attach"),
            confidence=confidence,
            classification_reasons=reasons,
            rejection_reason=reasons[0] if reasons else None,
        )
        classified.append(item)
    return classified


def infer_contact_mode(
    candidates: list[dict],
    sample_count: int,
    fs: float,
    config: CascadeStateConfig,
) -> dict:
    """由独立检出的边沿推断掩码；状态永不删除或重分类边沿。"""
    updated = [dict(item) for item in sorted(candidates, key=lambda x: x.get("center_index", 0))]
    accepted = []
    for item in updated:
        kind = item.get("candidate_type")
        item["accepted"] = kind in {"detach", "attach"}
        if item["accepted"]:
            item["event_type"] = kind
            item["rejection_reason"] = None
            accepted.append(item)
    initial_contact = not accepted or accepted[0]["event_type"] == "detach"

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
        if end < start:
            # 两个已检出边沿若局部区间重叠，事件仍保留；掩码仅使用可用采样。
            continue
        if current_contact:
            contact[cursor:start] = True
            weight[cursor:start] = 1.0
        else:
            noncontact[cursor:start] = True
        transition[start : end + 1] = True
        target_contact = event["event_type"] == "attach"
        weight[start : end + 1] = np.linspace(
            float(current_contact), float(target_contact), end - start + 1
        )
        current_contact = target_contact
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
    measured = []
    for index, candidate in enumerate(coarse):
        center = int(candidate["center_index"])
        lower = 0 if index == 0 else (int(coarse[index - 1]["center_index"]) + center) // 2 + 1
        upper = raw.size - 1 if index + 1 == len(coarse) else (
            center + int(coarse[index + 1]["center_index"])
        ) // 2
        measured.append(measure_edge_morphology(
            clean, candidate, fs, config, edge_diagnostics["reference_noise"],
            (lower, upper),
        ))
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
        "derivative_score": edge_diagnostics["derivative_score"],
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
