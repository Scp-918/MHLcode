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
    uh_peak_to_coarse_mean_sigma: float = 20.0
    uh_peak_period_fraction: float = 0.75
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
    extrema_trace_sec: float = 1.0
    threshold_dupeak: float = 0.15
    threshold_du: float = 2.0
    threshold_du_left: float = 2.0
    threshold_dupeak_left: float = 0.15
    threshold_du_right: float = 2.0
    threshold_dupeak_right: float = 0.08
    core_end_min_points: int = 2
    sharp_edge_duration_sec: float = 0.05
    deriv_peak_width_threshold_sec: float = 0.04
    deriv_peak_sigma_k: float = 5.0
    min_amplitude_sigma_k: float = 2.0
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
    """Find motion-only positive peaks and one ROI per distance-selected main peak."""
    values = np.asarray(edge_signal, dtype=float)
    prominence = config.uh_peak_prominence_sigma * sigma_uh
    coarse_mean_rise_threshold = config.uh_peak_to_coarse_mean_sigma * sigma_uh
    if f0_hz <= 0:
        raise ValueError("f0_hz must be positive")
    motion_start = motion_interval["start_idx"]
    motion_end = motion_interval["end_idx"]
    segment = values[motion_start:motion_end]
    distance_points = max(1, int(np.ceil(config.uh_peak_period_fraction * fs / f0_hz)))
    main_indices, _ = find_peaks(segment, prominence=prominence, plateau_size=1,
                                 distance=distance_points)
    main_indices = set(main_indices + motion_start)
    peaks = []
    indices, props = find_peaks(segment, prominence=prominence, plateau_size=1)
    for position, local_peak_idx in enumerate(indices):
        peak_idx = int(local_peak_idx + motion_start)
        coarse_left_idx = int(props["left_bases"][position] + motion_start)
        coarse_right_idx = int(props["right_bases"][position] + motion_start)
        coarse_mean_rise = float(values[peak_idx] -
                                 (values[coarse_left_idx] + values[coarse_right_idx]) / 2)
        if coarse_mean_rise <= coarse_mean_rise_threshold:
            continue
        peaks.append({"peak_idx": int(peak_idx), "sign": 1,
                      "is_main": peak_idx in main_indices, "roi_id": None,
                      "prominence": float(props["prominences"][position]),
                      "coarse_mean_rise": coarse_mean_rise,
                      "coarse_left_idx": coarse_left_idx,
                      "coarse_right_idx": coarse_right_idx,
                      "plateau_left_idx": int(props["left_edges"][position] + motion_start),
                      "plateau_right_idx": int(props["right_edges"][position] + motion_start)})
    peaks.sort(key=lambda row: row["peak_idx"])
    main_peaks = [peak for peak in peaks if peak["is_main"]]
    flat_points = max(1, round(config.uh_flat_min_sec * fs))
    for position, peak in enumerate(main_peaks):
        left_limit = peak["coarse_left_idx"]
        right_limit = peak["coarse_right_idx"]
        if position:
            left_limit = max(left_limit, (main_peaks[position-1]["peak_idx"] + peak["peak_idx"]) // 2)
        if position + 1 < len(main_peaks):
            right_limit = min(right_limit, (peak["peak_idx"] + main_peaks[position+1]["peak_idx"]) // 2)
        peak["refined_left_idx"] = _refined_boundary(
            values, peak["plateau_left_idx"], left_limit, -1, peak["sign"],
            config.uh_flat_slope_sigma * sigma_duh, flat_points,
            config.uh_peak_prominence_sigma * sigma_uh)
        peak["refined_right_idx"] = _refined_boundary(
            values, peak["plateau_right_idx"], right_limit, 1, peak["sign"],
            config.uh_flat_slope_sigma * sigma_duh, flat_points,
            config.uh_peak_prominence_sigma * sigma_uh)
    rois = _main_peak_rois(values, main_peaks, peaks, fs=fs, f0_hz=f0_hz)
    lead_points = round(config.udeco_left_lead_sec * fs)
    for roi in rois:
        roi["search_left_idx"] = max(0, roi["left_idx"] - lead_points)
    return rois, peaks


def _main_peak_rois(values: np.ndarray, main_peaks: list[dict], peaks: list[dict], *,
                    fs: float, f0_hz: float) -> list[dict]:
    """Cover nearby peak cycles and leave long gaps outside the ROIs."""
    rois = [{"roi_id": f"ROI_{i}", "left_idx": peak["refined_left_idx"],
             "right_idx": peak["refined_right_idx"], "main_peak_idx": peak["peak_idx"],
             "peak_indices": [], "source_peaks": [], "split_right_idx": None}
            for i, peak in enumerate(main_peaks)]
    for earlier, later in zip(rois, rois[1:]):
        start, end = earlier["main_peak_idx"], later["main_peak_idx"]
        short_gap = (end - start) / fs <= 1.5 / f0_hz
        if short_gap or earlier["right_idx"] >= later["left_idx"]:
            valley = start + 1 + int(np.argmin(values[start + 1:end]))
            earlier["right_idx"] = valley
            earlier["split_right_idx"] = valley
            later["left_idx"] = valley + 1
    for peak in peaks:
        for roi in rois:
            if roi["left_idx"] <= peak["peak_idx"] <= roi["right_idx"]:
                peak["roi_id"] = roi["roi_id"]
                roi["peak_indices"].append(peak["peak_idx"])
                roi["source_peaks"].append(peak)
                break
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
    lo, hi = roi.get("search_left_idx", roi["left_idx"]), roi["right_idx"]
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


def _derivative_extrema_candidates(score: np.ndarray, roi: dict) -> list[dict]:
    """Return the strongest signed sample difference of each polarity in one ROI."""
    lo = roi.get("search_left_idx", roi["left_idx"])
    hi = roi["right_idx"]
    segment = score[lo:hi+1]
    candidates = []
    for direction in (1, -1):
        center = lo + int(np.argmax(direction * segment))
        if direction * score[center] <= 0:
            continue
        candidates.append({"roi_id": roi["roi_id"], "center_idx": center,
                           "region_start_idx": center, "region_end_idx": center,
                           "direction": direction,
                           "detector_score": float(abs(score[center])),
                           "detector_name": "derivative_extrema"})
    return candidates


def detect_roi_candidates(edge_signal: np.ndarray, rois: list[dict], *, fs: float,
                          config: ExperimentConfig) -> tuple[list[dict], dict]:
    """Find candidates within Uh ROIs, normalized only by first-15-s noise."""
    values = np.asarray(edge_signal, dtype=float)
    baseline_n = round(config.baseline_sec * fs)
    if len(values) < baseline_n or baseline_n < 3:
        raise ValueError("recording must contain the complete baseline interval")
    if config.edge_detector == "derivative_extrema":
        derivative = np.zeros(len(values), dtype=float)
        derivative[1:] = np.diff(values)
        sigma_du = float(np.std(np.diff(values[:baseline_n])))
        score = derivative / max(sigma_du, np.finfo(float).eps)
        candidates = [candidate for roi in rois
                      for candidate in _derivative_extrema_candidates(score, roi)]
        diagnostics = {"derivative_signal": derivative, "derivative_score": score,
                       "derivative_sigma": sigma_du, "detector_name": config.edge_detector}
        return sorted(candidates, key=lambda item: (item["roi_id"], item["center_idx"])), diagnostics
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
    return {**candidate, **measurements, "resolved": False, "reason": reason,
            "endpoint_source": "threshold", "search_limit": False}


def localize_roi_candidates(clean: np.ndarray, edge_signal: np.ndarray,
                            candidates: list[dict], rois: list[dict], *, fs: float,
                            noise: dict[str, float], config: ExperimentConfig) -> list[dict]:
    """Locate each complete edge on clean-minus-context; no plateau is required."""
    edge_signal = np.asarray(edge_signal, dtype=float)
    derivative = np.diff(edge_signal)
    roi_by_id = {roi["roi_id"]: roi for roi in rois}
    radius = max(1, round(config.localize_search_sec * fs))
    neighbor_bounds = {}
    for roi_id in roi_by_id:
        ordered = sorted((i for i, row in enumerate(candidates) if row["roi_id"] == roi_id),
                         key=lambda i: candidates[i]["center_idx"])
        for position, i in enumerate(ordered):
            center = int(candidates[i]["center_idx"])
            if candidates[i]["detector_name"] == "derivative_extrema":
                trace_radius = max(radius, round(config.extrema_trace_sec * fs))
                outer_left = max(0, center - trace_radius)
                outer_right = min(len(edge_signal) - 1, center + trace_radius)
            else:
                outer_left = max(0, roi_by_id[roi_id].get("search_left_idx", roi_by_id[roi_id]["left_idx"])
                                 - radius)
                outer_right = min(len(edge_signal) - 1, roi_by_id[roi_id]["right_idx"] + radius)
            left = ((int(candidates[ordered[position-1]]["center_idx"]) + center) // 2
                    if position else outer_left)
            right = ((center + int(candidates[ordered[position+1]]["center_idx"])) // 2
                     if position + 1 < len(ordered) else outer_right)
            neighbor_bounds[i] = (left, right)
    result = []
    for candidate_index, candidate in enumerate(candidates):
        roi = roi_by_id[candidate["roi_id"]]
        search_left, search_right = neighbor_bounds[candidate_index]
        center = int(candidate["center_idx"])
        direction = int(candidate["direction"])
        if candidate["detector_name"] == "derivative_extrema":
            trace_radius = max(radius, round(config.extrema_trace_sec * fs))
            left = max(0, center - trace_radius)
            right = min(len(edge_signal) - 1, center + trace_radius)
        else:
            left = max(0, roi.get("search_left_idx", roi["left_idx"]) - radius)
            right = min(len(edge_signal) - 1, roi["right_idx"] + radius)
        lo = max(search_left, center - radius - 1)
        hi = min(search_right - 1, center + radius)
        if hi < lo:
            result.append(_unresolved(candidate, "no_derivative_window"))
            continue
        # Extrema discovery already fixed the derivative peak inside the ROI.
        # Do not move its anchor to a stronger transition outside that ROI.
        peak_k = (center - 1 if candidate["detector_name"] == "derivative_extrema"
                  else lo + int(np.argmax(direction * derivative[lo:hi+1])))
        peak = float(derivative[peak_k])
        if direction * peak < config.threshold_du * noise["sigma_du"]:
            result.append(_unresolved(candidate, "weak_derivative",
                                      derivative_peak_idx=peak_k + 1,
                                      derivative_peak=peak))
            continue
        extrema = candidate["detector_name"] == "derivative_extrema"
        threshold_left = max((config.threshold_du_left if extrema else config.threshold_du) * noise["sigma_du"],
                             (config.threshold_dupeak_left if extrema else config.threshold_dupeak) * abs(peak))
        threshold_right = max((config.threshold_du_right if extrema else config.threshold_du) * noise["sigma_du"],
                              (config.threshold_dupeak_right if extrema else config.threshold_dupeak) * abs(peak))
        core_threshold = threshold_left
        start_idx = None
        end_idx = None
        start_rule = None
        end_rule = None
        if (extrema and peak_k >= left + 2 and peak_k + 2 < right
                and 0 < direction * derivative[peak_k - 1] < threshold_left
                and 0 < direction * derivative[peak_k - 2] < threshold_left
                and abs(derivative[peak_k + 1]) < threshold_right
                and abs(derivative[peak_k + 2]) < threshold_right):
            start_idx, end_idx = peak_k - 2, peak_k + 1
            start_rule = end_rule = "single_point"
        low_run = 0
        if start_idx is None:
            for k in range(peak_k - 1, left - 1, -1):
                if extrema and direction * derivative[k] < 0:
                    start_idx, start_rule = k + 1, "reversal"
                    break
                low_run = low_run + 1 if abs(derivative[k]) < threshold_left else 0
                if low_run >= config.core_end_min_points:
                    start_idx, start_rule = k + low_run, "threshold"
                    break
        if start_idx is None:
            result.append(_unresolved(candidate, "core_does_not_start",
                                      derivative_peak_idx=peak_k + 1,
                                      derivative_peak=peak, core_threshold=core_threshold,
                                      threshold_left=threshold_left, threshold_right=threshold_right))
            continue
        low_run = 0
        reverse_count = 0
        if end_idx is None:
            for k in range(peak_k + 1, right):
                if extrema and direction * derivative[k] < 0:
                    reverse_count += 1
                    if reverse_count >= 2:
                        end_idx, end_rule = k, "second_reversal"
                        break
                low_run = low_run + 1 if abs(derivative[k]) < threshold_right else 0
                if low_run >= config.core_end_min_points:
                    end_idx, end_rule = k - low_run + 1, "threshold"
                    break
        partial = {"start_idx": start_idx, "end_idx": end_idx,
                   "derivative_peak_idx": peak_k + 1, "derivative_peak": peak,
                   "core_threshold": core_threshold,
                   "threshold_left": threshold_left, "threshold_right": threshold_right,
                   "start_rule": start_rule, "end_rule": end_rule}
        if end_idx is None:
            result.append(_unresolved(candidate, "core_does_not_end", **partial))
            continue
        if end_idx <= start_idx:
            result.append(_unresolved(candidate, "invalid_edge_bounds", **partial))
            continue
        half_peak = abs(peak) / 2
        fwhm_left = peak_k
        while fwhm_left > left and direction * derivative[fwhm_left-1] >= half_peak:
            fwhm_left -= 1
        fwhm_right = peak_k
        while fwhm_right < right-1 and direction * derivative[fwhm_right+1] >= half_peak:
            fwhm_right += 1
        delta = float(edge_signal[end_idx] - edge_signal[start_idx])
        signed_steps = direction * derivative[start_idx:end_idx]
        result.append({**candidate, **partial, "resolved": True, "reason": None,
                       "endpoint_source": "threshold", "search_limit": False,
                       "delta": delta, "amplitude": abs(delta),
                       "T_edge": (end_idx - start_idx) / fs,
                       "normalized_derivative_peak": abs(peak) / max(noise["sigma_du"], np.finfo(float).eps),
                       "derivative_peak_width": (fwhm_right - fwhm_left + 1) / fs,
                       "derivative_fwhm_left_idx": fwhm_left,
                       "derivative_fwhm_right_idx": fwhm_right + 1,
                       "monotonicity": float(np.mean(signed_steps >= -config.threshold_du * noise["sigma_du"])),
                       "baseline_snr": abs(delta) / max(noise["sigma_udeco"], np.finfo(float).eps)})
    _split_extrema_overlaps(result, edge_signal, fs=fs, noise=noise, config=config)
    return result


def _split_extrema_overlaps(rows, edge_signal, *, fs, noise, config):
    """Split overlapping opposite extrema at the quietest derivative sample."""
    derivative = np.diff(edge_signal, prepend=edge_signal[0])
    changed = set()
    for roi_id in dict.fromkeys(row["roi_id"] for row in rows):
        pair = sorted((i for i, row in enumerate(rows)
                       if row["roi_id"] == roi_id
                       and row["detector_name"] == "derivative_extrema"),
                      key=lambda i: rows[i]["center_idx"])
        if len(pair) != 2 or rows[pair[0]]["direction"] == rows[pair[1]]["direction"]:
            continue
        i, j = pair
        first, second = rows[i], rows[j]
        end = first.get("end_idx")
        start = second.get("start_idx")
        if end is None or start is None or end < start:
            continue
        lo, hi = first["center_idx"] + 1, second["center_idx"]
        if lo >= hi:
            continue
        split = lo + int(np.argmin(np.abs(derivative[lo:hi])))
        for row in (first, second):
            row["overlap_split_idx"] = split
        if split < end:
            first["end_idx"] = split
            first["end_rule"] = "overlap_split"
        if split + 1 > start:
            second["start_idx"] = split + 1
            second["start_rule"] = "overlap_split"
        changed.update(pair)
    for i in changed:
        row = rows[i]
        if not row["resolved"]:
            continue
        start, end = row["start_idx"], row["end_idx"]
        if end <= start:
            row.update(resolved=False, reason="invalid_edge_bounds_after_split")
            continue
        delta = float(edge_signal[end] - edge_signal[start])
        row.update(delta=delta, amplitude=abs(delta), T_edge=(end-start)/fs,
                   baseline_snr=abs(delta)/max(noise["sigma_udeco"], np.finfo(float).eps),
                   monotonicity=float(np.mean(row["direction"] * derivative[start+1:end+1]
                                              >= -config.threshold_du * noise["sigma_du"])))


def _fallback_boundary(values: np.ndarray, peak: int, limit: int, step: int) -> tuple[int, str]:
    """Walk away from a signal peak until a growing opposite derivative appears."""
    edge_point = peak
    original_sign = 0
    previous_size = 0.0
    for point in range(peak + step, limit + step, step):
        difference = values[point + 1] - values[point] if step < 0 else values[point] - values[point - 1]
        sign = int(np.sign(difference))
        size = abs(difference)
        if sign and original_sign and sign != original_sign and size > previous_size:
            return edge_point, "turn"
        if sign and not original_sign:
            original_sign = sign
        edge_point = point
        previous_size = size
    return edge_point, "search_limit"


def recover_missing_extrema_pairs(edge_signal: np.ndarray, localized: list[dict], rois: list[dict],
                                  *, fs: float, noise: dict[str, float],
                                  config: ExperimentConfig) -> list[dict]:
    """Append two peak-centered edges only where extrema lack a localized pair."""
    values = np.asarray(edge_signal, dtype=float)
    derivative = np.diff(values, prepend=values[0])
    result = list(localized)
    reach = max(1, round(config.extrema_trace_sec * fs))
    for roi in rois:
        original = [row for row in localized if row["roi_id"] == roi["roi_id"]
                    and row["detector_name"] == "derivative_extrema"]
        if {row["direction"] for row in original if row["resolved"]} == {1, -1}:
            continue
        left = roi.get("search_left_idx", roi["left_idx"])
        right = roi["right_idx"]
        segment = values[left:right + 1]
        peak = left + int(np.argmax(np.abs(segment - np.mean(segment))))
        limits = (max(0, peak - reach), min(len(values) - 1, peak + reach))
        for side, step, limit in (("left", -1, limits[0]), ("right", 1, limits[1])):
            bound, source = _fallback_boundary(values, peak, limit, step)
            start, end = (bound, peak) if step < 0 else (peak, bound)
            delta = float(values[end] - values[start])
            direction = int(np.sign(delta))
            base = {"roi_id": roi["roi_id"], "center_idx": peak,
                    "region_start_idx": start, "region_end_idx": end,
                    "detector_name": "peak_fallback", "fallback_peak_idx": peak,
                    "fallback_side": side, "endpoint_source": source,
                    "search_limit": source == "search_limit",
                    "detector_score": abs(values[peak]) / max(noise["sigma_udeco"], np.finfo(float).eps)}
            if end <= start or not direction:
                result.append({**base, "direction": direction, "resolved": False,
                               "reason": "invalid_fallback_edge", "start_idx": start, "end_idx": end})
                continue
            local_derivative = derivative[start + 1:end + 1]
            relative_peak = int(np.argmax(np.abs(local_derivative)))
            peak_idx = start + 1 + relative_peak
            derivative_peak = float(derivative[peak_idx])
            half_peak = abs(derivative_peak) / 2
            width_left = peak_idx
            while width_left > start + 1 and direction * derivative[width_left - 1] >= half_peak:
                width_left -= 1
            width_right = peak_idx
            while width_right < end and direction * derivative[width_right + 1] >= half_peak:
                width_right += 1
            result.append({**base, "direction": direction, "resolved": True, "reason": None,
                           "start_idx": start, "end_idx": end, "delta": delta,
                           "amplitude": abs(delta), "T_edge": (end - start) / fs,
                           "derivative_peak_idx": peak_idx, "derivative_peak": derivative_peak,
                           "core_threshold": max(config.threshold_du * noise["sigma_du"],
                                                 config.threshold_dupeak * abs(derivative_peak)),
                           "normalized_derivative_peak": abs(derivative_peak) / max(noise["sigma_du"], np.finfo(float).eps),
                           "derivative_peak_width": (width_right - width_left + 1) / fs,
                           "derivative_fwhm_left_idx": width_left,
                           "derivative_fwhm_right_idx": width_right,
                           "monotonicity": float(np.mean(direction * local_derivative
                                                          >= -config.threshold_du * noise["sigma_du"])),
                           "baseline_snr": abs(delta) / max(noise["sigma_udeco"], np.finfo(float).eps)})
    return result


def build_event_table(localized: list[dict], rois: list[dict], *, channel: str,
                      sigma_udeco: float, fs: float, config: ExperimentConfig,
                      direction_map: dict[int, dict[str, str]]) -> pd.DataFrame:
    """Retain every candidate and select one ordered left/peak edge pair per ROI."""
    rows = []
    for item in localized:
        row = {**item, "channel": channel, "is_main_rising": False,
               "is_main_falling": False, "morphology": None, "event_label": None,
               "edge_role": None, "pair_status": "missing_pair",
               "overlap_duplicate": False, "duplicate_owner_roi_id": None}
        if item.get("resolved"):
            shape = "sharp" if item["T_edge"] <= config.sharp_edge_duration_sec else "gradual"
            row["morphology"] = shape
            row["derivative_support"] = (item["derivative_peak_width"] <= config.deriv_peak_width_threshold_sec
                                         and item["normalized_derivative_peak"] >= config.deriv_peak_sigma_k)
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=["channel", "roi_id", "center_idx", "direction", "delta",
                                     "resolved", "is_main_rising", "is_main_falling",
                                     "morphology", "event_label", "edge_role", "pair_status",
                                     "overlap_duplicate"])
    table["edge_position_idx"] = (table["start_idx"].fillna(table["center_idx"])
                                  if "start_idx" in table else table["center_idx"])
    duplicate_points = round(config.candidate_merge_sec * fs)
    for earlier, later in zip(rois, rois[1:]):
        later_search_left = later.get("search_left_idx", later["left_idx"])
        earlier_rows = table.loc[(table.roi_id == earlier["roi_id"])
                                 & (table.center_idx >= later_search_left)]
        later_rows = table.loc[table.roi_id == later["roi_id"]]
        for index, row in earlier_rows.iterrows():
            matching = later_rows.loc[(later_rows.direction == row.direction)
                                      & ((later_rows.center_idx - row.center_idx).abs()
                                         <= duplicate_points)]
            if not matching.empty:
                table.at[index, "overlap_duplicate"] = True
                table.at[index, "duplicate_owner_roi_id"] = later["roi_id"]
    for roi in rois:
        eligible = table.loc[(table.roi_id == roi["roi_id"])
                             & (table.resolved == True)
                             & (table.overlap_duplicate == False)
                             & (table.direction * table.delta
                                >= config.min_amplitude_sigma_k * sigma_udeco)]
        left = roi["left_idx"]
        peak = roi["main_peak_idx"]
        right = roi["right_idx"]
        peak_zone = eligible.loc[(eligible.center_idx >= (left + peak) // 2)
                                 & (eligible.center_idx <= right)]
        peak_zone = peak_zone.reindex(peak_zone.delta.abs().sort_values(ascending=False).index)
        for peak_index, peak_edge in peak_zone.iterrows():
            left_zone = eligible.loc[(eligible.center_idx
                                      >= roi.get("search_left_idx", left))
                                     & (eligible.center_idx <= peak)
                                     & (eligible.center_idx < peak_edge.center_idx)
                                     & (eligible.direction == -peak_edge.direction)]
            if left_zone.empty:
                continue
            left_index = left_zone.delta.abs().idxmax()
            for index, role in ((left_index, "left"), (peak_index, "peak")):
                direction = int(table.at[index, "direction"])
                table.at[index, "is_main_rising" if direction > 0 else "is_main_falling"] = True
                table.at[index, "edge_role"] = role
                shape = "sharp" if table.at[index, "T_edge"] <= config.sharp_edge_duration_sec else "gradual"
                table.at[index, "event_label"] = direction_map[direction][shape]
            table.loc[table.roi_id == roi["roi_id"], "pair_status"] = "paired"
            break
    return table


def build_direct_event_table(localized: list[dict], rois: list[dict], *, channel: str,
                             sigma_udeco: float, fs: float, config: ExperimentConfig,
                             direction_map: dict[int, dict[str, str]]) -> pd.DataFrame:
    """Keep detected edges without Uh position-pair filtering."""
    directions = {roi["roi_id"]: set() for roi in rois}
    for item in localized:
        directions[item["roi_id"]].add(item["direction"])
    rows = []
    for item in localized:
        direction = int(item["direction"])
        resolved = bool(item["resolved"])
        amplitude_pass = resolved and abs(item["delta"]) > config.min_amplitude_sigma_k * sigma_udeco
        shape = ("sharp" if item["T_edge"] <= config.sharp_edge_duration_sec else "gradual") if amplitude_pass else None
        rows.append({**item, "channel": channel,
                     "is_main_rising": direction > 0, "is_main_falling": direction < 0,
                     "edge_role": ("roi_rising" if direction > 0 else
                                   "roi_falling" if direction < 0 else "fallback_unresolved"),
                     "pair_status": ("both_detected" if directions[item["roi_id"]] == {1, -1}
                                     else "missing_direction"),
                     "amplitude_pass": bool(amplitude_pass),
                     "classification_status": ("classified" if amplitude_pass else
                                               "low_amplitude" if resolved else "localization_failed"),
                     "direction_delta_conflict": bool(resolved and direction * item["delta"] < 0),
                     "morphology": shape,
                     "event_label": direction_map[direction][shape] if amplitude_pass else None,
                     "derivative_support": (item["derivative_peak_width"] <= config.deriv_peak_width_threshold_sec
                                            and item["normalized_derivative_peak"] >= config.deriv_peak_sigma_k)
                                            if resolved else None,
                     "main_source": None, "fallback_used": False,
                     "overlap_duplicate": False, "duplicate_owner_roi_id": None})
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=["channel", "roi_id", "center_idx", "direction",
                                     "resolved", "is_main_rising", "is_main_falling",
                                     "classification_status", "direction_delta_conflict"])
    for roi in rois:
        selected = table.roi_id == roi["roi_id"]
        extrema = selected & (table.detector_name == "derivative_extrema")
        fallback = selected & (table.detector_name == "peak_fallback")
        if not extrema.any() and not fallback.any():
            continue
        original_pair = set(table.loc[extrema & table.resolved, "direction"]) == {1, -1}
        fallback_pair = set(table.loc[fallback & table.resolved, "direction"]) == {1, -1}
        use_fallback = not original_pair and fallback_pair
        table.loc[selected, ["is_main_rising", "is_main_falling"]] = False
        main = fallback if use_fallback else extrema
        table.loc[main & (table.direction > 0), "is_main_rising"] = True
        table.loc[main & (table.direction < 0), "is_main_falling"] = True
        table.loc[selected, "main_source"] = ("peak_fallback" if use_fallback else
                                               "derivative_extrema" if extrema.any() else None)
        table.loc[selected, "fallback_used"] = use_fallback
        table.loc[selected, "pair_status"] = ("fallback_localized" if use_fallback else
                                               "original_localized" if original_pair else "unpaired")
    for earlier, later in zip(rois, rois[1:]):
        later_left = later.get("search_left_idx", later["left_idx"])
        earlier_rows = table.loc[(table.roi_id == earlier["roi_id"])
                                 & (table.center_idx >= later_left)]
        later_rows = table.loc[table.roi_id == later["roi_id"]]
        for index, row in earlier_rows.iterrows():
            duplicate = later_rows.loc[(later_rows.direction == row.direction)
                                       & ((later_rows.center_idx - row.center_idx).abs()
                                          <= round(config.candidate_merge_sec * fs))]
            if not duplicate.empty:
                table.at[index, "overlap_duplicate"] = True
                table.at[index, "duplicate_owner_roi_id"] = later["roi_id"]
    return table
