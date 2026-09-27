"""Uh-guided, baseline-normalized Udeco edge experiment.

The detector and the morphology measurement are deliberately separate. Uh
supplies intervals only; every Udeco edge is discovered within its paired ROI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks

from udeco_analysis import (AnalysisConfig, _repair_pydisplay_sequence,
                            detect_motion_stage, load_udeco_sample)
from udeco_edge_detection import (CascadeStateConfig, estimate_context_trend,
                                  median_clean, merge_candidate_fragments)


@dataclass(frozen=True)
class ExperimentConfig:
    median_kernel: int = 4
    context_window_sec: float = 10.0
    context_robust_iters: int = 1
    baseline_sec: float = 15.0
    uh_peak_prominence_sigma: float = 4.0
    uh_peak_to_coarse_l_sigma: float = 10.0
    uh_peak_period_fraction: float = 0.5
    uh_flat_slope_sigma: float = 2.0
    uh_flat_min_sec: float = 0.05
    fft_min_hz: float = 0.5
    fft_max_hz: float = 5.0
    udeco_left_lead_sec: float = 0.03
    edge_detector: str = "median_diff"
    median_diff_windows: tuple[int, ...] = (1, 3, 5, 7)
    edge_score_sigma: float = 3.5
    candidate_merge_sec: float = 0.02
    localize_search_sec: float = 0.20
    pre_plateau_sec: float = 0.05
    post_plateau_sec: float = 0.05
    plateau_sigma_k: float = 2.5
    deriv_core_ratio: float = 0.20
    deriv_sigma_k: float = 3.0
    core_end_min_points: int = 2
    settle_window_sec: float = 0.07
    settle_deriv_sigma_k: float = 2.5
    settle_hold_sec: float = 0.05
    sharp_core_duration_sec: float = 0.05
    deriv_peak_width_threshold_sec: float = 0.04
    deriv_peak_sigma_k: float = 5.0
    long_tail_sec: float = 0.10
    min_amplitude_sigma_k: float = 3.0
    haar_scales_sec: tuple[float, ...] = (0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56)
    haar_single_scale_sigma: float = 3.5
    haar_derivative_sigma: float = 3.5
    haar_min_scale_support: int = 2
    haar_merge_sec: float = 0.08
    haar_fragment_merge_sec: float = 0.02
    haar_fragment_min_spearman: float = 0.50
    haar_fragment_min_forward_fraction: float = 0.60


def load_paired_sample(sensor_csv: str | Path, *, data_root: str | Path, fs: int) -> dict:
    """Read repaired Uh2/Uh3 beside the existing loader's Udeco2/Udeco3."""
    loaded = load_udeco_sample(sensor_csv, data_root=data_root, fs=fs)
    repaired, _, _ = _repair_pydisplay_sequence(pd.read_csv(sensor_csv))
    uh = {}
    for name in ("Uh2", "Uh3"):
        values = pd.to_numeric(repaired[name], errors="coerce")
        uh[name] = values.interpolate(limit_direction="both").to_numpy(dtype=float)
    signals = {"Uh2": uh["Uh2"], "Udeco2": loaded.udeco2,
               "Uh3": uh["Uh3"], "Udeco3": loaded.udeco3}
    if any(len(values) != len(loaded.time_s) for values in signals.values()):
        raise ValueError("Uh and Udeco repaired sample axes disagree")
    return {"sample_name": loaded.sample_name, "subject": loaded.subject,
            "time_s": loaded.time_s, "fs": loaded.fs, "signals": signals,
            "acc": {"ACC_X": loaded.accx, "ACC_Y": loaded.accy, "ACC_Z": loaded.accz},
            "metadata": loaded.metadata}


def resolve_motion_interval(sample: dict, *, baseline_sec: float,
                            manual_start_sec: float | None = None,
                            manual_end_sec: float | None = None) -> dict:
    """Return one half-open motion interval, with optional manual bounds."""
    fs = sample["fs"]
    if (manual_start_sec is None) != (manual_end_sec is None):
        raise ValueError("set both manual motion start and end, or neither")
    if manual_start_sec is None:
        acc = sample["acc"]
        result = detect_motion_stage(acc["ACC_X"], acc["ACC_Y"], acc["ACC_Z"],
                                     fs=fs, config=AnalysisConfig(motion_calibration_sec=baseline_sec))
        if result["status"] != "ok":
            raise ValueError(f"ACC motion detection failed: {result['reason']}; set manual motion bounds")
        start_sec, end_sec = result["motion_start_s"], result["motion_end_s"]
        source = "acc"
    else:
        start_sec, end_sec = manual_start_sec, manual_end_sec
        source = "manual"
    left, right = round(start_sec * fs), round(end_sec * fs)
    if not 0 <= left < right <= len(sample["time_s"]):
        raise ValueError("motion interval must lie inside the recording")
    return {"start_idx": left, "end_idx": right, "start_sec": left / fs,
            "end_sec": right / fs, "source": source}


def estimate_motion_frequency(edge_signal: np.ndarray, *, fs: float,
                              motion_interval: dict, min_hz: float = 0.5,
                              max_hz: float = 5.0,
                              override_hz: float | None = None) -> dict:
    """Use the strongest Hann-window FFT bin as the motion-frequency candidate."""
    values = np.asarray(edge_signal, dtype=float)
    start, end = motion_interval["start_idx"], motion_interval["end_idx"]
    segment = values[start:end] - np.mean(values[start:end])
    if len(segment) < 3:
        raise ValueError("motion interval is too short for an FFT")
    frequencies = np.fft.rfftfreq(len(segment), d=1 / fs)
    power = np.abs(np.fft.rfft(segment * np.hanning(len(segment)))) ** 2
    band = (frequencies >= min_hz) & (frequencies <= max_hz)
    if not np.any(band):
        raise ValueError("FFT band has no frequency bins")
    band_indices = np.flatnonzero(band)
    dominant = int(band_indices[np.argmax(power[band])])
    if power[dominant] <= 0 and override_hz is None:
        raise ValueError("motion Uh spectrum has no usable peak; set an f0 override")
    if override_hz is not None and override_hz <= 0:
        raise ValueError("f0 override must be positive")
    return {"f0_hz": float(override_hz if override_hz is not None else frequencies[dominant]),
            "fft_peak_hz": float(frequencies[dominant]), "frequencies_hz": frequencies,
            "power": power, "source": "manual" if override_hz is not None else "fft"}


def preprocess_channel(raw: np.ndarray, fs: float, config: ExperimentConfig) -> dict:
    """Apply the existing short median and LOWESS Bcontext implementation."""
    prior = CascadeStateConfig(
        median_filter_points=config.median_kernel,
        context_window_s=config.context_window_sec,
        context_robust_iterations=config.context_robust_iters,
    )
    clean = median_clean(raw, prior)
    context = estimate_context_trend(clean, fs, prior)
    return {"raw": np.asarray(raw, dtype=float), "clean": clean,
            "Bcontext": context["Bcontext"], "edge_signal": context["edge_signal"]}


def baseline_sigmas(uh_edge_signal: np.ndarray, udeco_edge_signal: np.ndarray,
                    *, fs: float, baseline_sec: float = 15.0) -> dict[str, float]:
    """Every noise scale comes only from the recording's fixed initial baseline."""
    count = round(baseline_sec * fs)
    uh = np.asarray(uh_edge_signal[:count], dtype=float)
    udeco = np.asarray(udeco_edge_signal[:count], dtype=float)
    if min(len(uh), len(udeco)) != count or count < 3:
        raise ValueError("recording must contain the complete baseline interval")
    return {"sigma_uh": float(np.std(uh)), "sigma_duh": float(np.std(np.diff(uh))),
            "sigma_udeco": float(np.std(udeco)),
            "sigma_du": float(np.std(np.diff(udeco)))}


def _refined_boundary(values: np.ndarray, start: int, stop: int, step: int,
                      sign: int, flat_slope: float, flat_points: int,
                      near_base_limit: float) -> int:
    """Walk away from a peak until a valley/crest or sustained flat slope."""
    signed = sign * values
    base_level = signed[stop]
    index = start
    while index != stop:
        nxt = index + step
        if nxt < 0 or nxt >= len(values):
            break
        near_base = signed[nxt] <= base_level + near_base_limit
        if (near_base and 0 < nxt < len(values) - 1
                and signed[nxt] <= min(signed[nxt-1], signed[nxt+1])):
            return nxt
        edge = min(index, nxt)
        if near_base and abs(values[nxt] - values[index]) <= flat_slope:
            lo = max(0, edge - flat_points + 1) if step < 0 else edge
            hi = edge + flat_points if step > 0 else edge + 1
            if hi <= len(values) - 1 and hi - lo >= flat_points and np.all(np.abs(np.diff(values[lo:hi+1])) <= flat_slope):
                return nxt
        index = nxt
    segment = signed[min(start, stop):max(start, stop)+1]
    return min(start, stop) + int(np.argmin(segment))


def find_uh_rois(edge_signal: np.ndarray, *, fs: float, sigma_uh: float,
                 sigma_duh: float, motion_interval: dict, f0_hz: float,
                 config: ExperimentConfig) -> tuple[list[dict], list[dict]]:
    """Find motion-only positive peaks and build period-aware, disjoint ROIs."""
    values = np.asarray(edge_signal, dtype=float)
    prominence = config.uh_peak_prominence_sigma * sigma_uh
    coarse_left_rise_threshold = config.uh_peak_to_coarse_l_sigma * sigma_uh
    if f0_hz <= 0:
        raise ValueError("f0_hz must be positive")
    motion_start = motion_interval["start_idx"]
    motion_end = motion_interval["end_idx"]
    peaks = []
    indices, props = find_peaks(values[motion_start:motion_end],
                                prominence=prominence, plateau_size=1)
    for position, local_peak_idx in enumerate(indices):
        peak_idx = int(local_peak_idx + motion_start)
        coarse_left_idx = int(props["left_bases"][position] + motion_start)
        coarse_left_rise = float(values[peak_idx] - values[coarse_left_idx])
        if coarse_left_rise <= coarse_left_rise_threshold:
            continue
        peaks.append({"peak_idx": int(peak_idx), "sign": 1,
                      "prominence": float(props["prominences"][position]),
                      "coarse_left_rise": coarse_left_rise,
                      "coarse_left_idx": coarse_left_idx,
                      "coarse_right_idx": int(props["right_bases"][position] + motion_start),
                      "plateau_left_idx": int(props["left_edges"][position] + motion_start),
                      "plateau_right_idx": int(props["right_edges"][position] + motion_start)})
    peaks.sort(key=lambda row: row["peak_idx"])
    flat_points = max(1, round(config.uh_flat_min_sec * fs))
    for position, peak in enumerate(peaks):
        left_limit = peak["coarse_left_idx"]
        right_limit = peak["coarse_right_idx"]
        if position:
            left_limit = max(left_limit, (peaks[position-1]["peak_idx"] + peak["peak_idx"]) // 2)
        if position + 1 < len(peaks):
            right_limit = min(right_limit, (peak["peak_idx"] + peaks[position+1]["peak_idx"]) // 2)
        peak["refined_left_idx"] = _refined_boundary(
            values, peak["plateau_left_idx"], left_limit, -1, peak["sign"],
            config.uh_flat_slope_sigma * sigma_duh, flat_points,
            config.uh_peak_prominence_sigma * sigma_uh)
        peak["refined_right_idx"] = _refined_boundary(
            values, peak["plateau_right_idx"], right_limit, 1, peak["sign"],
            config.uh_flat_slope_sigma * sigma_duh, flat_points,
            config.uh_peak_prominence_sigma * sigma_uh)
    rois = _group_uh_peaks(values, peaks, fs=fs, period_fraction=config.uh_peak_period_fraction,
                           f0_hz=f0_hz)
    return rois, peaks


def _group_uh_peaks(values: np.ndarray, peaks: list[dict], *, fs: float,
                    period_fraction: float, f0_hz: float) -> list[dict]:
    """Group sub-period peaks, then split overlapping groups at their valley."""
    groups: list[list[dict]] = []
    max_gap_sec = period_fraction / f0_hz
    for peak in peaks:
        if groups and (peak["peak_idx"] - groups[-1][-1]["peak_idx"]) / fs < max_gap_sec:
            groups[-1].append(peak)
        else:
            groups.append([peak])
    rois = []
    for group in groups:
        main_peak = max(group, key=lambda item: item["coarse_left_rise"])
        rois.append({"roi_id": f"ROI_{len(rois)}",
                     "left_idx": min(item["refined_left_idx"] for item in group),
                     "right_idx": max(item["refined_right_idx"] for item in group),
                     "main_peak_idx": main_peak["peak_idx"],
                     "peak_indices": [item["peak_idx"] for item in group],
                     "source_peaks": group, "split_right_idx": None})
    for earlier, later in zip(rois, rois[1:]):
        if earlier["right_idx"] >= later["left_idx"]:
            start = earlier["peak_indices"][-1]
            end = later["peak_indices"][0]
            valley = start + int(np.argmin(values[start:end + 1]))
            earlier["right_idx"] = valley
            earlier["split_right_idx"] = valley
            later["left_idx"] = valley + 1
    return rois


def _window_change(values: np.ndarray, width: int) -> np.ndarray:
    """Right median minus left median at a sample boundary; width 1 is diff."""
    result = np.zeros(len(values), dtype=float)
    if 2 * width > len(values):
        return result
    medians = np.median(sliding_window_view(values, width), axis=1)
    centers = np.arange(width, len(values) - width + 1)
    result[centers] = medians[centers] - medians[centers - width]
    return result


def _detector_scales(values: np.ndarray, widths: list[tuple[int, int | float]],
                     baseline_n: int) -> tuple[dict, dict, dict]:
    responses, scores, sigmas = {}, {}, {}
    for width, label in widths:
        response = _window_change(values, width)
        baseline = response[width:baseline_n-width+1]
        sigma = float(np.std(baseline))
        responses[label] = response
        scores[label] = response / max(sigma, np.finfo(float).eps)
        sigmas[label] = sigma
    return responses, scores, sigmas


def _best_signed_score(scale_scores: dict, n: int) -> np.ndarray:
    best = np.zeros(n, dtype=float)
    for row in scale_scores.values():
        stronger = np.abs(row) > np.abs(best)
        best[stronger] = row[stronger]
    return best


def _score_regions(score: np.ndarray, roi: dict, threshold: float,
                   merge_points: int) -> list[np.ndarray]:
    lo, hi = roi["left_idx"], roi["right_idx"]
    active = np.arange(lo, hi + 1)
    active = active[np.abs(score[active]) >= threshold]
    if not len(active):
        return []
    groups = [[int(active[0])]]
    for index in active[1:]:
        previous = groups[-1][-1]
        if index - previous <= merge_points and np.sign(score[index]) == np.sign(score[previous]):
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    return [np.asarray(group, dtype=int) for group in groups]


def _median_candidates(scores: dict, combined: np.ndarray, roi: dict,
                       fs: float, config: ExperimentConfig) -> list[dict]:
    regions = _score_regions(combined, roi, config.edge_score_sigma,
                             max(1, round(config.candidate_merge_sec * fs)))
    candidates = []
    for indices in regions:
        center = int(indices[np.argmax(np.abs(combined[indices]))])
        candidates.append({"roi_id": roi["roi_id"], "center_idx": center,
                           "region_start_idx": int(indices[0]),
                           "region_end_idx": int(indices[-1]),
                           "direction": int(np.sign(combined[center])),
                           "detector_score": float(abs(combined[center])),
                           "detector_name": "median_diff"})
    return candidates


def _haar_candidates(values: np.ndarray, scores: dict, combined: np.ndarray,
                     derivative_score: np.ndarray, roi: dict, fs: float,
                     config: ExperimentConfig) -> list[dict]:
    primitive = []
    groups = _score_regions(combined, roi, config.edge_score_sigma,
                            max(1, round(config.haar_merge_sec * fs)))
    for indices in groups:
        center = int(indices[np.argmax(np.abs(combined[indices]))])
        direction = int(np.sign(combined[center]))
        support = [scale for scale, row in scores.items()
                   if np.any(direction * row[indices] >= config.edge_score_sigma)]
        peak_score = float(max(direction * scores[s][indices].max() if direction > 0
                               else -scores[s][indices].min() for s in support))
        if len(support) < config.haar_min_scale_support and peak_score < config.haar_single_scale_sigma:
            continue
        fine_scale = min(support)
        fine_indices = indices[np.argmax(direction * scores[fine_scale][indices])]
        primitive.append({"center_index": int(fine_indices),
                          "coarse_start_index": int(indices[0]),
                          "coarse_end_index": int(indices[-1]),
                          "best_scale_s": float(fine_scale),
                          "edge_score": direction * peak_score,
                          "noise_scale": 1.0,
                          "support_scales_s": tuple(support),
                          "detection_channels": ("haar",)})
    derivative_groups = _score_regions(derivative_score, roi,
                                       config.haar_derivative_sigma, 1)
    for indices in derivative_groups:
        center = int(indices[np.argmax(np.abs(derivative_score[indices]))])
        primitive.append({"center_index": center,
                          "coarse_start_index": int(indices[0]),
                          "coarse_end_index": int(indices[-1]),
                          "best_scale_s": 1.0 / fs,
                          "edge_score": float(derivative_score[center]),
                          "noise_scale": 1.0, "support_scales_s": (),
                          "detection_channels": ("derivative",)})
    merge_config = CascadeStateConfig(
        edge_fragment_merge_s=config.haar_fragment_merge_sec,
        edge_fragment_min_spearman=config.haar_fragment_min_spearman,
        edge_fragment_min_forward_fraction=config.haar_fragment_min_forward_fraction)
    merged = merge_candidate_fragments(values, primitive, fs, merge_config)
    return [{"roi_id": roi["roi_id"], "center_idx": int(row["center_index"]),
             "region_start_idx": int(row["coarse_start_index"]),
             "region_end_idx": int(row["coarse_end_index"]),
             "direction": int(np.sign(row["edge_score"])),
             "detector_score": float(abs(row["edge_score"])),
             "detector_name": "haar",
             "fragment_centers": row["fragment_centers"]} for row in merged]


def detect_roi_candidates(edge_signal: np.ndarray, rois: list[dict], *, fs: float,
                          config: ExperimentConfig) -> tuple[list[dict], dict]:
    """Find candidates within Uh ROIs, normalized only by first-15-s noise."""
    values = np.asarray(edge_signal, dtype=float)
    baseline_n = round(config.baseline_sec * fs)
    if len(values) < baseline_n or baseline_n < 3:
        raise ValueError("recording must contain the complete baseline interval")
    if config.edge_detector == "median_diff":
        widths = [(int(width), int(width)) for width in config.median_diff_windows]
    elif config.edge_detector == "haar":
        widths = [(max(2, round(scale * fs)), float(scale))
                  for scale in config.haar_scales_sec if 2 * round(scale * fs) <= len(values)]
    else:
        raise ValueError(f"unknown EDGE_DETECTOR: {config.edge_detector}")
    responses, scores, sigmas = _detector_scales(values, widths, baseline_n)
    combined = _best_signed_score(scores, len(values))
    diagnostics = {"scale_responses": responses, "scale_scores": scores,
                   "scale_sigmas": sigmas, "combined_score": combined,
                   "threshold": config.edge_score_sigma,
                   "detector_name": config.edge_detector}
    candidates = []
    if config.edge_detector == "haar":
        derivative = np.zeros(len(values), dtype=float)
        difference = np.diff(values)
        sigma_du = float(np.std(difference[:baseline_n-1]))
        derivative[1:] = difference / max(sigma_du, np.finfo(float).eps)
        diagnostics["derivative_score"] = derivative
    for roi in rois:
        if config.edge_detector == "median_diff":
            candidates.extend(_median_candidates(scores, combined, roi, fs, config))
        else:
            candidates.extend(_haar_candidates(values, scores, combined, derivative,
                                               roi, fs, config))
    return sorted(candidates, key=lambda item: (item["roi_id"], item["center_idx"])), diagnostics


def _unresolved(candidate: dict, reason: str, **measurements) -> dict:
    return {**candidate, **measurements, "resolved": False, "reason": reason}


def localize_roi_candidates(clean: np.ndarray, edge_signal: np.ndarray,
                            candidates: list[dict], rois: list[dict], *, fs: float,
                            noise: dict[str, float], config: ExperimentConfig) -> list[dict]:
    """Measure each candidate on Udeco; detector region width is never duration."""
    clean = np.asarray(clean, dtype=float)
    edge_signal = np.asarray(edge_signal, dtype=float)
    derivative = np.diff(edge_signal)
    roi_by_id = {roi["roi_id"]: roi for roi in rois}
    neighbor_bounds = {}
    for roi_id in roi_by_id:
        ordered = sorted((i for i, row in enumerate(candidates) if row["roi_id"] == roi_id),
                         key=lambda i: candidates[i]["center_idx"])
        for position, i in enumerate(ordered):
            center = int(candidates[i]["center_idx"])
            left = (int(candidates[ordered[position-1]]["center_idx"]) + center) // 2 if position else roi_by_id[roi_id]["left_idx"]
            right = (center + int(candidates[ordered[position+1]]["center_idx"])) // 2 if position + 1 < len(ordered) else roi_by_id[roi_id]["right_idx"]
            neighbor_bounds[i] = (left, right)
    radius = max(1, round(config.localize_search_sec * fs))
    pre_points = max(1, round(config.pre_plateau_sec * fs))
    post_points = max(1, round(config.post_plateau_sec * fs))
    settle_points = max(1, round(config.settle_window_sec * fs))
    hold_points = max(1, round(config.settle_hold_sec * fs))
    result = []
    for candidate_index, candidate in enumerate(candidates):
        roi = roi_by_id[candidate["roi_id"]]
        left, right = int(roi["left_idx"]), int(roi["right_idx"])
        search_left, search_right = neighbor_bounds[candidate_index]
        center = int(candidate["center_idx"])
        direction = int(candidate["direction"])
        lo = max(search_left, center - radius - 1)
        hi = min(search_right - 1, center + radius)
        if hi < lo:
            result.append(_unresolved(candidate, "no_derivative_window"))
            continue
        peak_k = lo + int(np.argmax(direction * derivative[lo:hi+1]))
        peak = float(derivative[peak_k])
        if direction * peak < config.deriv_sigma_k * noise["sigma_du"]:
            result.append(_unresolved(candidate, "weak_derivative",
                                      derivative_peak_idx=peak_k + 1,
                                      derivative_peak=peak))
            continue
        core_threshold = max(config.deriv_sigma_k * noise["sigma_du"],
                             config.deriv_core_ratio * abs(peak))
        start_idx = peak_k
        while start_idx > left and direction * derivative[start_idx-1] >= core_threshold:
            start_idx -= 1
        core_end_idx = None
        low_run = 0
        for k in range(peak_k + 1, right):
            low_run = low_run + 1 if abs(derivative[k]) < core_threshold else 0
            if low_run >= config.core_end_min_points:
                core_end_idx = k - low_run + 1
                break
        partial = {"start_idx": start_idx, "core_end_idx": core_end_idx,
                   "derivative_peak_idx": peak_k + 1, "derivative_peak": peak,
                   "core_threshold": core_threshold}
        if core_end_idx is None:
            result.append(_unresolved(candidate, "core_does_not_end", **partial))
            continue
        if start_idx - pre_points + 1 < left:
            result.append(_unresolved(candidate, "missing_pre_plateau", **partial))
            continue
        pre_values = clean[start_idx-pre_points+1:start_idx+1]
        pre_level = float(np.median(pre_values))
        if np.median(np.abs(pre_values - pre_level)) > config.plateau_sigma_k * noise["sigma_udeco"]:
            result.append(_unresolved(candidate, "unstable_pre_plateau", **partial))
            continue
        settle_limit = config.settle_deriv_sigma_k * noise["sigma_du"]
        stable_run = 0
        stable_start = None
        settle_end_idx = None
        for k in range(core_end_idx, right - settle_points + 1):
            stable = np.median(np.abs(derivative[k:k+settle_points])) < settle_limit
            if stable:
                stable_run += 1
                if stable_start is None:
                    stable_start = k
                if stable_run >= hold_points:
                    settle_end_idx = stable_start
                    break
            else:
                stable_run = 0
                stable_start = None
        if settle_end_idx is None:
            result.append(_unresolved(candidate, "not_settled_in_roi", **partial))
            continue
        partial["settle_end_idx"] = settle_end_idx
        if settle_end_idx + post_points > right:
            result.append(_unresolved(candidate, "missing_post_plateau", **partial))
            continue
        post_values = clean[settle_end_idx+1:settle_end_idx+1+post_points]
        post_level = float(np.median(post_values))
        if np.median(np.abs(post_values - post_level)) > config.plateau_sigma_k * noise["sigma_udeco"]:
            result.append(_unresolved(candidate, "unstable_post_plateau", **partial))
            continue
        half_peak = abs(peak) / 2
        fwhm_left = peak_k
        while fwhm_left > left and direction * derivative[fwhm_left-1] >= half_peak:
            fwhm_left -= 1
        fwhm_right = peak_k
        while fwhm_right < right-1 and direction * derivative[fwhm_right+1] >= half_peak:
            fwhm_right += 1
        delta = post_level - pre_level
        signed_core_steps = direction * derivative[start_idx:core_end_idx]
        result.append({**candidate, **partial, "resolved": True, "reason": None,
                       "pre_level": pre_level, "post_level": post_level,
                       "delta": delta, "amplitude": abs(delta),
                       "T_core": (core_end_idx - start_idx) / fs,
                       "T_tail": (settle_end_idx - core_end_idx) / fs,
                       "normalized_derivative_peak": abs(peak) / max(noise["sigma_du"], np.finfo(float).eps),
                       "derivative_peak_width": (fwhm_right - fwhm_left + 1) / fs,
                       "derivative_fwhm_left_idx": fwhm_left,
                       "derivative_fwhm_right_idx": fwhm_right + 1,
                       "monotonicity": float(np.mean(signed_core_steps >= -config.deriv_sigma_k * noise["sigma_du"])),
                       "baseline_snr": abs(delta) / max(noise["sigma_udeco"], np.finfo(float).eps)})
    return result


def build_event_table(localized: list[dict], rois: list[dict], *, channel: str,
                      sigma_udeco: float, config: ExperimentConfig,
                      direction_map: dict[int, dict[str, str]]) -> pd.DataFrame:
    """Keep all candidates but designate only the largest rise/fall per ROI."""
    rows = []
    for item in localized:
        row = {**item, "channel": channel, "is_main_rising": False,
               "is_main_falling": False, "morphology": None, "event_label": None}
        if item.get("resolved"):
            shape = "sharp" if item["T_core"] <= config.sharp_core_duration_sec else "gradual"
            row["morphology"] = ("sharp_long_tail" if shape == "sharp"
                                  and item["T_tail"] >= config.long_tail_sec else shape)
            row["derivative_support"] = (item["derivative_peak_width"] <= config.deriv_peak_width_threshold_sec
                                         and item["normalized_derivative_peak"] >= config.deriv_peak_sigma_k)
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=["channel", "roi_id", "center_idx", "direction", "delta",
                                     "resolved", "is_main_rising", "is_main_falling",
                                     "morphology", "event_label"])
    for roi in rois:
        for direction, column in ((1, "is_main_rising"), (-1, "is_main_falling")):
            eligible = table.loc[(table.roi_id == roi["roi_id"])
                                 & (table.resolved == True)
                                 & (table.direction == direction)
                                 & (direction * table.delta >= config.min_amplitude_sigma_k * sigma_udeco)]
            if eligible.empty:
                continue
            selected = eligible.delta.abs().idxmax()
            table.at[selected, column] = True
            shape = "sharp" if table.at[selected, "T_core"] <= config.sharp_core_duration_sec else "gradual"
            table.at[selected, "event_label"] = direction_map[direction][shape]
    return table
