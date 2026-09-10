import numpy as np

from udeco_cascade_state_decomposition import (
    CascadeStateConfig,
    classify_edge_candidates,
    detect_multiscale_edges,
    estimate_context_trend,
    measure_edge_morphology,
    median_clean,
)


def _edge_pair_signal(fs: float = 100.0) -> tuple[np.ndarray, int, int]:
    """构造快上升、短平台和慢下降组成的接触模态切换。"""
    time = np.arange(int(12 * fs)) / fs
    signal = 0.02 + 2e-5 * time
    detach = int(4.0 * fs)
    attach = int(6.0 * fs)
    signal[detach : detach + 3] += np.linspace(0.0, 0.012, 3)
    signal[detach + 3 :] += 0.012
    signal[attach : attach + 20] -= np.linspace(0.0, 0.012, 20)
    signal[attach + 20 :] -= 0.012
    return signal, detach, attach


def test_four_point_median_removes_spike_but_keeps_persistent_step():
    signal = np.r_[np.zeros(20), np.ones(20)]
    signal[8] = 9.0

    clean = median_clean(signal, CascadeStateConfig())

    assert clean[8] == 0.0
    assert np.all(clean[24:] == 1.0)


def test_morphology_classifies_fast_up_and_slower_down_edges():
    fs = 100.0
    signal, detach_index, attach_index = _edge_pair_signal(fs)
    config = CascadeStateConfig(edge_z_threshold=3.0)
    clean = median_clean(signal, config)
    context = estimate_context_trend(clean, fs, config)
    coarse, diagnostics = detect_multiscale_edges(
        clean, context["Bcontext"], fs, config
    )
    measured = [
        measure_edge_morphology(
            diagnostics["edge_signal"], candidate, fs, config,
            diagnostics["reference_noise"],
        )
        for candidate in coarse
    ]
    candidates = classify_edge_candidates(measured, config)

    detach = [
        item for item in candidates
        if item["candidate_type"] == "detach"
        and abs(item["center_index"] - detach_index) <= 8
    ]
    attach = [
        item for item in candidates
        if item["candidate_type"] == "attach"
        and abs(item["center_index"] - attach_index) <= 15
    ]

    assert detach and detach[0]["edge_width_points"] <= 4
    assert attach and attach[0]["edge_width_points"] >= 5
