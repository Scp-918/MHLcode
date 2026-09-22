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


def test_short_median_removes_spike_but_keeps_persistent_step():
    signal = np.r_[np.zeros(20), np.ones(20)]
    signal[8] = 9.0

    clean = median_clean(signal, CascadeStateConfig())

    assert clean[8] == 0.0
    assert np.all(clean[24:] == 1.0)


def test_default_cleaner_uses_three_points_for_a_fast_edge():
    """奇数短窗去掉孤立尖峰，不给两点快速模态段引入偶数窗偏移。"""
    signal = np.zeros(32)
    signal[8] = 9.0
    signal[16:18] = 1.0

    clean = median_clean(signal, CascadeStateConfig())

    assert CascadeStateConfig().median_filter_points == 3
    assert clean[8] == 0.0
    np.testing.assert_array_equal(clean[15:19], [0.0, 1.0, 1.0, 0.0])


def test_haar_candidates_record_consistent_scale_support():
    """真实阶跃的相邻尺度响应应共同支持候选，而非只留下单一最高分。"""
    fs = 100.0
    signal = np.full(800, 0.02)
    signal[400:] += 0.01
    config = CascadeStateConfig()
    coarse, _ = detect_multiscale_edges(signal, np.zeros_like(signal), fs, config)
    nearby = [item for item in coarse if abs(item["center_index"] - 400) <= 8]

    assert nearby
    assert nearby[0]["scale_support"] >= 2
    assert len(nearby[0]["support_scales_s"]) == nearby[0]["scale_support"]


def test_morphology_uses_final_platforms_for_full_edge_boundaries():
    """start/end 对应离开 L、稳定进入 R，而不是 10%/90% 穿越点。"""
    signal = np.r_[np.zeros(100), np.linspace(0.0, 0.01, 6), np.full(100, 0.01)]
    candidate = {
        "center_index": 103,
        "best_scale_s": 0.04,
        "edge_score": 8.0,
        "noise_scale": 1e-5,
        "reference_level": 0.02,
    }

    edge = measure_edge_morphology(signal, candidate, 100.0, CascadeStateConfig(), 1e-5)

    assert edge["start_index"] == 101
    assert edge["end_index"] == 105
    assert edge["edge_width_points"] == edge["end_index"] - edge["start_index"]
    assert edge["edge_width_10_90_points"] >= edge["edge_width_points"]
    assert np.isclose(edge["platform_tolerance"], 0.0003)


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


def test_classification_records_each_failed_morphology_gate():
    """多个形态门槛同时不满足时，诊断信息不能只剩笼统的类别。"""
    base = {
        "level_snr": 2.0,
        "relative_amplitude": 0.02,
        "persistence": 0.4,
        "monotonicity": 0.8,
        "morphology_noise": 0.001,
        "level_left": 0.0,
        "level_right": 0.01,
        "level_delta": 0.01,
        "edge_score": 5.0,
    }
    slow_up = {**base, "direction": "up", "edge_width_points": 7}
    classified = classify_edge_candidates([slow_up], CascadeStateConfig())

    assert classified[0]["candidate_type"] == "pressure_like"
    assert classified[0]["classification_reasons"] == (
        "detach_width_too_long",
        "detach_low_snr",
        "detach_small_amplitude",
        "detach_short_persistence",
    )


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
