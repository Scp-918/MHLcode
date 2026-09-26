"""Uh-guided, baseline-normalized Udeco edge experiment.

The detector and the morphology measurement are deliberately separate. Uh
supplies intervals only; every Udeco edge is discovered within its paired ROI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from udeco_analysis import _repair_pydisplay_sequence, load_udeco_sample
from udeco_edge_detection import CascadeStateConfig, estimate_context_trend, median_clean


@dataclass(frozen=True)
class ExperimentConfig:
    median_kernel: int = 4
    context_window_sec: float = 10.0
    context_robust_iters: int = 1
    baseline_sec: float = 15.0
    uh_peak_prominence_sigma: float = 4.0
    uh_peak_distance_sec: float = 0.30
    uh_flat_slope_sigma: float = 2.0
    uh_flat_min_sec: float = 0.05
    uh_roi_margin_sec: float = 0.15
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
            "metadata": loaded.metadata}


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
                      sign: int, flat_slope: float, flat_points: int) -> int:
    """Walk away from a peak until a valley/crest or sustained flat slope."""
    signed = sign * values
    index = start
    while index != stop:
        nxt = index + step
        if nxt < 0 or nxt >= len(values):
            break
        if 0 < nxt < len(values) - 1 and signed[nxt] <= min(signed[nxt-1], signed[nxt+1]):
            return nxt
        edge = min(index, nxt)
        if abs(values[nxt] - values[index]) <= flat_slope:
            lo = max(0, edge - flat_points + 1) if step < 0 else edge
            hi = edge + flat_points if step > 0 else edge + 1
            if hi <= len(values) - 1 and hi - lo >= flat_points and np.all(np.abs(np.diff(values[lo:hi+1])) <= flat_slope):
                return nxt
        index = nxt
    segment = signed[min(start, stop):max(start, stop)+1]
    return min(start, stop) + int(np.argmin(segment))


def find_uh_rois(edge_signal: np.ndarray, *, fs: float, sigma_uh: float,
                 sigma_duh: float, config: ExperimentConfig) -> tuple[list[dict], list[dict]]:
    """Detect signed Uh peaks, refine their boundaries, and merge ROI overlaps."""
    values = np.asarray(edge_signal, dtype=float)
    prominence = config.uh_peak_prominence_sigma * sigma_uh
    distance = max(1, round(config.uh_peak_distance_sec * fs))
    peaks = []
    for sign in (1, -1):
        indices, props = find_peaks(sign * values, prominence=prominence,
                                    distance=distance, plateau_size=1)
        for position, peak_idx in enumerate(indices):
            if sign * values[peak_idx] < prominence:
                continue
            peaks.append({"peak_idx": int(peak_idx), "sign": sign,
                          "prominence": float(props["prominences"][position]),
                          "coarse_left_idx": int(props["left_bases"][position]),
                          "coarse_right_idx": int(props["right_bases"][position]),
                          "plateau_left_idx": int(props["left_edges"][position]),
                          "plateau_right_idx": int(props["right_edges"][position])})
    peaks.sort(key=lambda row: row["peak_idx"])
    flat_points = max(1, round(config.uh_flat_min_sec * fs))
    margin = round(config.uh_roi_margin_sec * fs)
    for position, peak in enumerate(peaks):
        left_limit = peak["coarse_left_idx"]
        right_limit = peak["coarse_right_idx"]
        if position:
            left_limit = max(left_limit, (peaks[position-1]["peak_idx"] + peak["peak_idx"]) // 2)
        if position + 1 < len(peaks):
            right_limit = min(right_limit, (peak["peak_idx"] + peaks[position+1]["peak_idx"]) // 2)
        peak["refined_left_idx"] = _refined_boundary(
            values, peak["plateau_left_idx"], left_limit, -1, peak["sign"],
            config.uh_flat_slope_sigma * sigma_duh, flat_points)
        peak["refined_right_idx"] = _refined_boundary(
            values, peak["plateau_right_idx"], right_limit, 1, peak["sign"],
            config.uh_flat_slope_sigma * sigma_duh, flat_points)
        peak["roi_left_idx"] = max(0, peak["refined_left_idx"] - margin)
        peak["roi_right_idx"] = min(len(values) - 1, peak["refined_right_idx"] + margin)
    rois = []
    for peak in sorted(peaks, key=lambda row: row["roi_left_idx"]):
        if rois and peak["roi_left_idx"] <= rois[-1]["right_idx"]:
            roi = rois[-1]
            roi["right_idx"] = max(roi["right_idx"], peak["roi_right_idx"])
            roi["peak_indices"].append(peak["peak_idx"])
            roi["source_peaks"].append(peak)
        else:
            rois.append({"roi_id": f"ROI_{len(rois)}", "left_idx": peak["roi_left_idx"],
                         "right_idx": peak["roi_right_idx"],
                         "peak_indices": [peak["peak_idx"]], "source_peaks": [peak]})
    return rois, peaks
