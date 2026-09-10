import numpy as np

from udeco_cascade_state_decomposition import (
    CascadeStateConfig,
    classify_edge_candidates,
    decompose_cascade_udeco,
    detect_multiscale_edges,
    estimate_context_trend,
    infer_contact_mode,
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


def _state_edge(center: int, start: int, end: int, kind: str) -> dict:
    """构造只包含状态机所需字段的高置信形态边沿。"""
    return {
        "center_index": center,
        "start_index": start,
        "end_index": end,
        "start_s": start / 100.0,
        "end_s": end / 100.0,
        "edge_width_points": end - start,
        "candidate_type": kind,
        "direction": "up" if kind == "detach" else "down",
        "confidence": 0.95,
        "accepted": False,
        "rejection_reason": "awaiting_state_inference",
    }


def test_independent_edges_create_a_short_noncontact_interval():
    config = CascadeStateConfig()
    candidates = [
        _state_edge(101, 100, 102, "detach"),
        _state_edge(132, 128, 136, "attach"),
    ]

    state = infer_contact_mode(candidates, 300, 100.0, config)

    assert [event["event_type"] for event in state["J_events"]] == ["detach", "attach"]
    assert state["noncontact_mask"][103:128].all()
    assert state["contact_mask"][:100].all()
    assert state["contact_mask"][137:].all()


def test_auto_state_supports_leading_and_trailing_noncontact():
    config = CascadeStateConfig()
    leading = infer_contact_mode(
        [_state_edge(52, 48, 56, "attach")], 120, 100.0, config
    )
    trailing = infer_contact_mode(
        [_state_edge(51, 50, 52, "detach")], 120, 100.0, config
    )

    assert leading["noncontact_mask"][:48].all()
    assert leading["contact_mask"][57:].all()
    assert trailing["contact_mask"][:50].all()
    assert trailing["noncontact_mask"][53:].all()


def test_slow_pressure_stays_contact_and_decomposition_closes():
    fs = 100.0
    time = np.arange(int(30 * fs)) / fs
    signal = 0.02 + 2e-5 * time
    loading = (time >= 8.0) & (time < 13.0)
    hold = (time >= 13.0) & (time < 18.0)
    relaxing = (time >= 18.0) & (time < 24.0)
    signal[loading] += 0.004 * (time[loading] - 8.0) / 5.0
    signal[hold] += 0.004
    signal[relaxing] += 0.004 * (1.0 - (time[relaxing] - 18.0) / 6.0)

    result = decompose_cascade_udeco(signal, fs, CascadeStateConfig())

    assert not result["noncontact_mask"].any()
    assert np.any(result["contact_substate"] != "baseline")
    np.testing.assert_allclose(
        result["clean"],
        result["J"] + result["P"] + result["Bslow"] + result["residual"],
    )
