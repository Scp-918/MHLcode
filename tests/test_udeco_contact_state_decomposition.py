"""接触状态分解的阶段一回归测试。"""

import numpy as np
import pytest

import udeco_contact_state_decomposition as contact_module
from udeco_contact_state_decomposition import (
    ContactStateConfig,
    decompose_contact_state_udeco,
    detect_transition_candidates,
    estimate_conditional_bslow,
    estimate_pilot_trend,
    hampel_filter,
    infer_contact_states,
    reconstruct_contact_mode,
)


def _synthetic_contact_signal(fs: float = 100.0) -> tuple[np.ndarray, int, int]:
    """生成包含慢漂移、压力坡、快脱离、慢附着和孤立尖峰的固定信号。"""
    rng = np.random.default_rng(20260910)
    time = np.arange(0.0, 18.0, 1.0 / fs)
    signal = 1.0 + 0.015 * time + 0.015 * np.sin(2 * np.pi * 0.25 * time)
    signal += np.clip((time - 2.0) / 3.0, 0.0, 1.0) * 0.35
    detach_start = int(7.0 * fs)
    detach_end = int(7.12 * fs)
    signal[detach_start:detach_end] += np.linspace(0.0, -0.65, detach_end - detach_start)
    signal[detach_end:] -= 0.65
    attach_start = int(11.0 * fs)
    attach_end = int(12.2 * fs)
    signal[attach_start:attach_end] += np.linspace(0.0, 0.65, attach_end - attach_start)
    signal[attach_end:] += 0.65
    signal += rng.normal(0.0, 0.01, size=time.size)
    signal[int(5.0 * fs)] += 4.0
    return signal, detach_start, attach_start


def _candidate(
    start: int,
    end: int,
    fs: float,
    *,
    event_type: str,
    direction: str,
    confidence: float = 0.95,
) -> dict:
    """构造与阶段一公开候选格式一致的确定性测试候选。"""
    return {
        "start_index": start,
        "end_index": end,
        "start_s": start / fs,
        "end_s": end / fs,
        "duration_s": (end - start) / fs,
        "delta": -6.0 if direction == "down" else 6.0,
        "direction": direction,
        "candidate_type": event_type,
        "peak_slope": 60.0,
        "peak_slope_z": 8.0,
        "monotonicity": 1.0,
        "confidence": confidence,
        "accepted": False,
        "rejection_reason": "awaiting_state_inference",
    }


def _use_known_candidates(monkeypatch, candidates: list[dict], bpilot: np.ndarray) -> None:
    """让阶段三测试聚焦状态之后的分解，不重复验证候选检测器。"""
    pilot = {
        "Bpilot": bpilot,
        "fast_residual": np.zeros_like(bpilot),
        "pilot_slope": np.gradient(bpilot),
    }
    monkeypatch.setattr(contact_module, "estimate_pilot_trend", lambda *_: pilot)
    monkeypatch.setattr(
        contact_module,
        "detect_transition_candidates",
        lambda *_: (candidates, {"residual": np.zeros_like(bpilot), "scores": np.zeros_like(bpilot), "scales_s": []}),
    )


def test_complete_detach_gap_attach_pair_is_accepted_and_inpainted():
    """若保守配对或锚点插值缺失，完整稳定的内部非接触段会被漏掉。"""
    fs = 20.0
    clean = np.full(int(42 * fs), 10.0)
    detach_start, detach_end = int(32 * fs), int(32.1 * fs)
    attach_start, attach_end = int(35 * fs), int(36 * fs)
    clean[detach_start : detach_end + 1] = np.linspace(10.0, 4.0, detach_end - detach_start + 1)
    clean[detach_end + 1 : attach_start] = 4.0
    clean[attach_start : attach_end + 1] = np.linspace(4.0, 10.0, attach_end - attach_start + 1)
    candidates = [
        _candidate(detach_start, detach_end, fs, event_type="detach", direction="down"),
        _candidate(attach_start, attach_end, fs, event_type="attach", direction="up"),
    ]

    state = infer_contact_states(clean, np.full_like(clean, 10.0), candidates, fs, ContactStateConfig())
    reconstruction = reconstruct_contact_mode(clean, state, fs, ContactStateConfig())

    assert [event["event_type"] for event in state["J_events"]] == ["detach", "attach"]
    assert all(candidate["accepted"] for candidate in state["candidates"])
    assert state["contact_mask"][: int(30 * fs)].all()
    assert state["noncontact_mask"][detach_end + 1 : attach_start].all()
    assert state["transition_mask"][detach_start : detach_end + 1].all()
    assert state["transition_mask"][attach_start : attach_end + 1].all()
    assert state["contact_weight"][detach_start] == pytest.approx(1.0)
    assert state["contact_weight"][detach_end] == pytest.approx(0.0)
    assert state["contact_weight"][attach_start] == pytest.approx(0.0)
    assert state["contact_weight"][attach_end] == pytest.approx(1.0)
    assert reconstruction["ucontact_synthetic_mask"][detach_start : attach_end + 1].all()
    assert reconstruction["Ucontact"][detach_start : attach_end + 1] == pytest.approx(10.0)


def test_dynamic_pressure_hold_remains_contact_and_uncertain():
    """若忽略间隙内的试探斜率活动，动态按压会被误判成非接触。"""
    fs = 20.0
    time = np.arange(int(42 * fs)) / fs
    clean = np.full(time.size, 10.0)
    detach_start, detach_end = int(32 * fs), int(32.1 * fs)
    attach_start, attach_end = int(35 * fs), int(36 * fs)
    hold = slice(detach_end + 1, attach_start)
    clean[detach_start : detach_end + 1] = np.linspace(10.0, 16.0, detach_end - detach_start + 1)
    clean[hold] = 16.0 + 0.8 * np.sin(2 * np.pi * 1.5 * time[hold])
    clean[attach_start : attach_end + 1] = np.linspace(16.0, 10.0, attach_end - attach_start + 1)
    bpilot = clean.copy()
    candidates = [
        _candidate(detach_start, detach_end, fs, event_type="detach", direction="up"),
        _candidate(attach_start, attach_end, fs, event_type="attach", direction="down"),
    ]

    state = infer_contact_states(clean, bpilot, candidates, fs, ContactStateConfig())

    assert not state["J_events"]
    assert state["contact_mask"].all()
    assert not state["noncontact_mask"].any()
    assert state["uncertain_mask"][detach_start : attach_end + 1].all()
    assert all(not candidate["accepted"] for candidate in state["candidates"])
    assert all(candidate["rejection_reason"] for candidate in state["candidates"])


def test_high_confidence_terminal_detach_uses_constant_contact_hold():
    """若终端脱离未获支持或外推斜率，稳定尾段的接触模式会丢失或漂移。"""
    fs = 20.0
    clean = np.full(int(40 * fs), 10.0)
    detach_start, detach_end = int(35 * fs), int(35.1 * fs)
    clean[detach_start : detach_end + 1] = np.linspace(10.0, 3.0, detach_end - detach_start + 1)
    clean[detach_end + 1 :] = 3.0
    candidates = [
        _candidate(
            detach_start,
            detach_end,
            fs,
            event_type="detach",
            direction="down",
            confidence=0.95,
        )
    ]

    config = ContactStateConfig()
    state = infer_contact_states(clean, clean, candidates, fs, config)
    reconstruction = reconstruct_contact_mode(clean, state, fs, config)

    assert [event["event_type"] for event in state["J_events"]] == ["detach"]
    assert state["candidates"][0]["accepted"] is True
    assert state["transition_mask"][detach_start : detach_end + 1].all()
    assert state["noncontact_mask"][detach_end + 1 :].all()
    assert reconstruction["ucontact_synthetic_mask"][detach_start:].all()
    assert reconstruction["Ucontact"][detach_start:] == pytest.approx(10.0)


def test_contact_recovery_ends_synthesis_without_cumulative_offset():
    """若恢复后继续合成或叠加锚点差，后续真实接触波形会产生累计偏移。"""
    fs = 20.0
    time = np.arange(int(42 * fs)) / fs
    clean = np.full(time.size, 10.0)
    detach_start, detach_end = int(32 * fs), int(32.1 * fs)
    attach_start, attach_end = int(34 * fs), int(35 * fs)
    clean[detach_start : detach_end + 1] = np.linspace(10.0, 4.0, detach_end - detach_start + 1)
    clean[detach_end + 1 : attach_start] = 4.0
    clean[attach_start : attach_end + 1] = np.linspace(4.0, 12.0, attach_end - attach_start + 1)
    clean[attach_end + 1 :] = 12.0 + 0.2 * np.sin(2 * np.pi * 0.5 * time[attach_end + 1 :])
    candidates = [
        _candidate(detach_start, detach_end, fs, event_type="detach", direction="down"),
        _candidate(attach_start, attach_end, fs, event_type="attach", direction="up"),
    ]

    config = ContactStateConfig()
    state = infer_contact_states(clean, np.full_like(clean, 10.0), candidates, fs, config)
    reconstruction = reconstruct_contact_mode(clean, state, fs, config)

    assert state["contact_mask"][attach_end + 1 :].all()
    assert reconstruction["ucontact_synthetic_mask"][detach_start : attach_end + 1].all()
    assert not reconstruction["ucontact_synthetic_mask"][attach_end + 1 :].any()
    np.testing.assert_array_equal(reconstruction["Ucontact"][attach_end + 1 :], clean[attach_end + 1 :])


def test_dynamic_contact_anchor_rejects_otherwise_complete_pair():
    """若接触锚窗不稳定仍接受配对，重建会把动态接触误当作锚点。"""
    fs = 20.0
    time = np.arange(int(42 * fs)) / fs
    clean = np.full(time.size, 10.0)
    detach_start, detach_end = int(32 * fs), int(32.1 * fs)
    attach_start, attach_end = int(35 * fs), int(36 * fs)
    before_anchor = slice(detach_start - int(fs), detach_start)
    clean[before_anchor] += 0.8 * np.sin(2 * np.pi * 2.0 * time[before_anchor])
    clean[detach_start : detach_end + 1] = np.linspace(10.0, 4.0, detach_end - detach_start + 1)
    clean[detach_end + 1 : attach_start] = 4.0
    clean[attach_start : attach_end + 1] = np.linspace(4.0, 10.0, attach_end - attach_start + 1)
    candidates = [
        _candidate(detach_start, detach_end, fs, event_type="detach", direction="down"),
        _candidate(attach_start, attach_end, fs, event_type="attach", direction="up"),
    ]

    state = infer_contact_states(clean, clean, candidates, fs, ContactStateConfig())
    reconstruction = reconstruct_contact_mode(clean, state, fs, ContactStateConfig())

    assert not state["J_events"]
    assert state["contact_mask"].all()
    assert state["uncertain_mask"][detach_start : attach_end + 1].all()
    assert {candidate["rejection_reason"] for candidate in state["candidates"]} == {
        "unstable_contact_anchor"
    }
    assert not reconstruction["ucontact_synthetic_mask"].any()
    np.testing.assert_array_equal(reconstruction["Ucontact"], clean)


def test_clean_residual_oscillation_rejects_pair_with_flat_pilot_anchor():
    """若只检查平坦 Bpilot，剧烈振荡的 clean 锚窗会被错误用于合成。"""
    fs = 20.0
    time = np.arange(int(42 * fs)) / fs
    clean = np.full(time.size, 10.0)
    bpilot = np.full_like(clean, 10.0)
    detach_start, detach_end = int(32 * fs), int(32.1 * fs)
    attach_start, attach_end = int(35 * fs), int(36 * fs)
    before_anchor = slice(detach_start - int(fs), detach_start)
    clean[before_anchor] += 3.0 * np.sin(2 * np.pi * 4.0 * time[before_anchor])
    clean[detach_start : detach_end + 1] = np.linspace(10.0, 4.0, detach_end - detach_start + 1)
    clean[detach_end + 1 : attach_start] = 4.0
    clean[attach_start : attach_end + 1] = np.linspace(4.0, 10.0, attach_end - attach_start + 1)
    candidates = [
        _candidate(detach_start, detach_end, fs, event_type="detach", direction="down"),
        _candidate(attach_start, attach_end, fs, event_type="attach", direction="up"),
    ]

    config = ContactStateConfig()
    state = infer_contact_states(clean, bpilot, candidates, fs, config)
    reconstruction = reconstruct_contact_mode(clean, state, fs, config)

    assert not state["J_events"]
    assert state["contact_mask"].all()
    assert state["uncertain_mask"][detach_start : attach_end + 1].all()
    assert {candidate["rejection_reason"] for candidate in state["candidates"]} == {
        "unstable_contact_anchor"
    }
    assert not reconstruction["ucontact_synthetic_mask"].any()
    np.testing.assert_array_equal(reconstruction["Ucontact"], clean)


def test_long_dynamic_gap_cannot_raise_its_own_pilot_activity_threshold():
    """若用整段 slope 校准，长动态会抬高阈值并把自身错误判成稳定 gap。"""
    fs = 20.0
    time = np.arange(int(140 * fs)) / fs
    bpilot = np.full(time.size, 10.0)
    detach_start, detach_end = int(32 * fs), int(32.1 * fs)
    attach_start, attach_end = int(130 * fs), int(131 * fs)
    gap = slice(detach_end + 1, attach_start)
    bpilot[gap] += np.sin(2 * np.pi * time[gap])
    clean = bpilot.copy()
    candidates = [
        _candidate(detach_start, detach_end, fs, event_type="detach", direction="down"),
        _candidate(attach_start, attach_end, fs, event_type="attach", direction="up"),
    ]

    state = infer_contact_states(clean, bpilot, candidates, fs, ContactStateConfig())

    assert not state["J_events"]
    assert state["contact_mask"].all()
    assert state["uncertain_mask"][detach_start : attach_end + 1].all()
    assert {candidate["rejection_reason"] for candidate in state["candidates"]} == {
        "excessive_pilot_slope_activity"
    }


@pytest.mark.parametrize(
    ("attach_offset", "attach_duration", "attach_direction"),
    [
        pytest.param(0.45, 1.0, "up", id="below-minimum-dwell"),
        pytest.param(2.0, 1.0, "down", id="not-opposite-direction"),
        pytest.param(2.0, 0.05, "up", id="attach-not-slower"),
    ],
)
def test_rejected_pair_marks_full_candidate_span_uncertain(
    attach_offset: float,
    attach_duration: float,
    attach_direction: str,
):
    """若配对门槛失败只标事件局部，中间模糊跨度会被错误视作稳定 Contact。"""
    fs = 20.0
    clean = np.full(int(40 * fs), 10.0)
    detach_start, detach_end = int(32 * fs), int(32.1 * fs)
    attach_start = detach_end + int(round(attach_offset * fs))
    attach_end = attach_start + int(round(attach_duration * fs))
    candidates = [
        _candidate(detach_start, detach_end, fs, event_type="detach", direction="down"),
        _candidate(
            attach_start,
            attach_end,
            fs,
            event_type="attach",
            direction=attach_direction,
        ),
    ]

    state = infer_contact_states(clean, clean, candidates, fs, ContactStateConfig())

    assert not state["J_events"]
    assert state["contact_mask"].all()
    assert not state["noncontact_mask"].any()
    assert state["uncertain_mask"][detach_start : attach_end + 1].all()
    assert all(candidate["rejection_reason"] for candidate in state["candidates"])


def test_pair_at_exact_half_second_dwell_is_accepted():
    """若 dwell 少算一个采样间隔，恰好 0.5 秒的合法下限会被误拒。"""
    fs = 20.0
    clean = np.full(int(40 * fs), 10.0)
    detach_start, detach_end = int(32 * fs), int(32.1 * fs)
    attach_start, attach_end = detach_end + int(0.5 * fs), detach_end + int(1.5 * fs)
    clean[detach_start : detach_end + 1] = np.linspace(10.0, 4.0, detach_end - detach_start + 1)
    clean[detach_end + 1 : attach_start] = 4.0
    clean[attach_start : attach_end + 1] = np.linspace(4.0, 10.0, attach_end - attach_start + 1)
    candidates = [
        _candidate(detach_start, detach_end, fs, event_type="detach", direction="down"),
        _candidate(attach_start, attach_end, fs, event_type="attach", direction="up"),
    ]

    state = infer_contact_states(clean, np.full_like(clean, 10.0), candidates, fs, ContactStateConfig())

    assert [event["event_type"] for event in state["J_events"]] == ["detach", "attach"]


def test_internal_pair_rejects_when_either_transition_has_low_confidence():
    """若只凭形态接受低置信转换，内部非接触区间会被过度推断。"""
    fs = 20.0
    clean = np.full(int(40 * fs), 10.0)
    detach_start, detach_end = int(32 * fs), int(32.1 * fs)
    attach_start, attach_end = int(34 * fs), int(35 * fs)
    candidates = [
        _candidate(
            detach_start,
            detach_end,
            fs,
            event_type="detach",
            direction="down",
            confidence=0.49,
        ),
        _candidate(attach_start, attach_end, fs, event_type="attach", direction="up"),
    ]

    state = infer_contact_states(clean, clean, candidates, fs, ContactStateConfig())

    assert not state["J_events"]
    assert state["contact_mask"].all()
    assert state["uncertain_mask"][detach_start : attach_end + 1].all()
    assert {candidate["rejection_reason"] for candidate in state["candidates"]} == {
        "low_transition_confidence"
    }


def test_state_inference_rejects_signal_too_short_for_gradient():
    """若不在接口处检查长度，调用者只会收到 NumPy 的底层 gradient 异常。"""
    with pytest.raises(ValueError, match="至少需要两个采样点"):
        infer_contact_states(
            np.array([1.0]),
            np.array([1.0]),
            [],
            20.0,
            ContactStateConfig(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_type", "other"),
        ("direction", "sideways"),
        ("duration_s", -0.1),
        ("duration_s", np.nan),
        ("confidence", 1.1),
        ("confidence", np.nan),
    ],
)
def test_state_inference_rejects_invalid_core_candidate_values(field, value):
    """若核心候选字段未校验，状态机可能静默接受无物理意义的转换。"""
    fs = 20.0
    candidate = _candidate(
        int(32 * fs),
        int(32.1 * fs),
        fs,
        event_type="detach",
        direction="down",
    )
    candidate[field] = value

    with pytest.raises(ValueError, match="候选"):
        infer_contact_states(
            np.full(int(40 * fs), 10.0),
            np.full(int(40 * fs), 10.0),
            [candidate],
            fs,
            ContactStateConfig(),
        )


def test_hampel_replaces_an_isolated_spike_with_local_value():
    """若尖峰没有被局部中位数替换，接触变化会被误报。"""
    signal, _, _ = _synthetic_contact_signal()
    clean = hampel_filter(signal, 100.0, ContactStateConfig())

    spike = int(5.0 * 100.0)
    assert clean[spike] == pytest.approx(np.median(clean[spike - 3 : spike + 4]), abs=0.03)
    assert abs(clean[spike] - signal[spike]) > 3.0


def test_pilot_trend_is_finite_and_matches_signal_length():
    """若试探趋势长度或数值错误，后续残差候选计算无法可靠执行。"""
    signal, _, _ = _synthetic_contact_signal()
    clean = hampel_filter(signal, 100.0, ContactStateConfig())
    pilot = estimate_pilot_trend(clean, 100.0, ContactStateConfig())

    assert set(pilot) == {"Bpilot", "fast_residual", "pilot_slope"}
    for values in pilot.values():
        assert values.shape == clean.shape
        assert np.isfinite(values).all()


def test_multiscale_candidates_include_fast_detach_and_slow_attach():
    """若多尺度合并或局部细化失效，会遗漏不同速度的真实转换。"""
    fs = 100.0
    signal, detach_start, attach_start = _synthetic_contact_signal(fs)
    config = ContactStateConfig()
    clean = hampel_filter(signal, fs, config)
    pilot = estimate_pilot_trend(clean, fs, config)
    candidates, diagnostics = detect_transition_candidates(clean, pilot["Bpilot"], fs, config)

    assert isinstance(diagnostics, dict)
    detach = [item for item in candidates if item["direction"] == "down" and abs(item["start_index"] - detach_start) < int(0.35 * fs)]
    attach = [item for item in candidates if item["direction"] == "up" and abs(item["start_index"] - attach_start) < int(0.5 * fs)]
    assert detach and attach
    assert detach[0]["duration_s"] > 1.0 / fs
    assert attach[0]["duration_s"] > 0.20
    assert detach[0]["candidate_type"] == "detach"
    assert attach[0]["candidate_type"] == "attach"
    required = {
        "start_index", "end_index", "start_s", "end_s", "duration_s", "delta",
        "direction", "candidate_type", "peak_slope", "peak_slope_z", "monotonicity",
        "confidence", "accepted", "rejection_reason",
    }
    assert required <= set(detach[0])
    assert detach[0]["accepted"] is False


@pytest.mark.parametrize(
    ("signal", "fs", "config"),
    [
        (np.array([]), 100.0, ContactStateConfig()),
        (np.ones((2, 2)), 100.0, ContactStateConfig()),
        (np.array([1.0, np.nan]), 100.0, ContactStateConfig()),
        (np.array([1.0, 2.0]), 0.0, ContactStateConfig()),
        (np.array([1.0, 2.0]), 100.0, "not-a-config"),
    ],
)
def test_public_functions_reject_invalid_signal_sampling_or_config(signal, fs, config):
    """若输入校验缺失，坏数据会在统计步骤中静默扩散。"""
    with pytest.raises((TypeError, ValueError)):
        hampel_filter(signal, fs, config)
    if isinstance(config, ContactStateConfig) and signal.ndim == 1 and signal.size:
        with pytest.raises((TypeError, ValueError)):
            estimate_pilot_trend(signal, fs, config)
        with pytest.raises((TypeError, ValueError)):
            detect_transition_candidates(signal, signal, fs, config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hampel_window_s": 0.0},
        {"pilot_lowess_window_s": 0.0},
        {"candidate_scales": (0.1, 0.0)},
        {"initial_contact_s": -1.0},
        {"state_transition_confidence": 1.5},
        {"candidate_scales": [0.05, 0.1]},
    ],
)
def test_config_rejects_invalid_detection_settings(kwargs):
    """若配置允许无效尺度或窗口，运行时检测将没有可解释的时间含义。"""
    with pytest.raises(ValueError):
        ContactStateConfig(**kwargs)


@pytest.mark.parametrize(
    "signal",
    [np.array([]), np.ones((2, 2)), np.array([1.0, np.inf])],
)
def test_estimate_pilot_trend_rejects_invalid_signal_directly(signal):
    """若趋势接口绕过输入校验，坏输入会进入 LOWESS。"""
    with pytest.raises((TypeError, ValueError)):
        estimate_pilot_trend(signal, 100.0, ContactStateConfig())


@pytest.mark.parametrize(
    "signal",
    [np.array([]), np.ones((2, 2)), np.array([1.0, np.inf])],
)
def test_candidate_detection_rejects_invalid_clean_input_directly(signal):
    """若候选接口绕过输入校验，坏输入会污染多尺度统计量。"""
    with pytest.raises((TypeError, ValueError)):
        detect_transition_candidates(signal, signal, 100.0, ContactStateConfig())


def test_complete_pair_produces_finite_closed_physical_components(monkeypatch):
    """完整脱离/附着后，P 在非接触段为零，J 恢复接触后也回到零。"""
    fs = 20.0
    time = np.arange(int(42 * fs)) / fs
    contact_level = 10.0 + 1e-4 * time
    signal = contact_level.copy()
    detach_start, detach_end = int(32 * fs), int(32.1 * fs)
    attach_start, attach_end = int(35 * fs), int(36 * fs)
    signal[detach_start : detach_end + 1] = np.linspace(
        contact_level[detach_start], 4.0, detach_end - detach_start + 1
    )
    signal[detach_end + 1 : attach_start] = 4.0
    signal[attach_start : attach_end + 1] = np.linspace(
        4.0, contact_level[attach_end], attach_end - attach_start + 1
    )
    candidates = [
        _candidate(detach_start, detach_end, fs, event_type="detach", direction="down"),
        _candidate(attach_start, attach_end, fs, event_type="attach", direction="up"),
    ]
    _use_known_candidates(monkeypatch, candidates, contact_level)

    config = ContactStateConfig(bslow_stable_offset_threshold=0.1)
    result = decompose_contact_state_udeco(signal, fs, config)

    required = {
        "raw", "clean", "Bpilot", "fast_residual", "pilot_slope",
        "transition_candidates", "candidate_score", "J_events", "contact_mask", "noncontact_mask",
        "transition_mask", "uncertain_mask", "contact_weight", "state_confidence",
        "Ucontact", "ucontact_synthetic_mask", "Bm_contact", "dBm_dt", "Bslow",
        "bslow_update_mask", "P", "P_valid_mask", "J", "mode_gap_raw", "residual",
    }
    assert required <= result.keys()
    assert [event["event_type"] for event in result["J_events"]] == ["detach", "attach"]
    assert np.all(result["P"][result["noncontact_mask"]] == 0.0)
    assert np.all(result["J"][attach_end + 1 :] == 0.0)
    assert not result["bslow_update_mask"][detach_start : attach_end + 1].any()
    assert np.ptp(result["Bslow"][detach_start - 1 : attach_end + 1]) == 0.0
    reconstruction = result["J"] + result["P"] + result["Bslow"] + result["residual"]
    np.testing.assert_allclose(reconstruction, result["clean"], atol=1e-12)
    for key in required - {"transition_candidates", "J_events"}:
        assert result[key].shape == signal.shape
        assert np.isfinite(result[key]).all()


def test_pressure_activity_remains_contact_and_is_not_moved_into_j(monkeypatch):
    """压力活动即使带来边沿候选，也应保持 Contact 并保留在连续 P 中。"""
    fs = 20.0
    time = np.arange(int(42 * fs)) / fs
    signal = np.full(time.size, 10.0)
    detach_start, detach_end = int(32 * fs), int(32.1 * fs)
    attach_start, attach_end = int(35 * fs), int(36 * fs)
    signal[detach_start : detach_end + 1] = np.linspace(
        10.0, 16.0, detach_end - detach_start + 1
    )
    gap = slice(detach_end + 1, attach_start)
    signal[gap] = 16.0 + 0.8 * np.sin(2 * np.pi * 1.5 * time[gap])
    signal[attach_start : attach_end + 1] = np.linspace(
        16.0, 10.0, attach_end - attach_start + 1
    )
    candidates = [
        _candidate(detach_start, detach_end, fs, event_type="detach", direction="up"),
        _candidate(attach_start, attach_end, fs, event_type="attach", direction="down"),
    ]
    _use_known_candidates(monkeypatch, candidates, signal)

    result = decompose_contact_state_udeco(signal, fs)

    assert not result["J_events"]
    assert not result["noncontact_mask"].any()
    assert result["uncertain_mask"][detach_start : attach_end + 1].all()
    assert np.all(result["J"] == 0.0)
    assert np.max(np.abs(result["P"])) > 1.0


def test_terminal_noncontact_freezes_bslow_and_uses_constant_ucontact(monkeypatch):
    """高置信末尾脱离后，接触模态保持最近锚点，慢基线不再更新。"""
    fs = 20.0
    signal = np.full(int(40 * fs), 10.0)
    detach_start, detach_end = int(35 * fs), int(35.1 * fs)
    signal[detach_start : detach_end + 1] = np.linspace(
        10.0, 3.0, detach_end - detach_start + 1
    )
    signal[detach_end + 1 :] = 3.0
    candidates = [
        _candidate(
            detach_start,
            detach_end,
            fs,
            event_type="detach",
            direction="down",
            confidence=0.95,
        )
    ]
    _use_known_candidates(monkeypatch, candidates, signal)

    result = decompose_contact_state_udeco(signal, fs)

    assert [event["event_type"] for event in result["J_events"]] == ["detach"]
    assert result["Ucontact"][detach_start:] == pytest.approx(10.0)
    assert np.all(result["P"][result["noncontact_mask"]] == 0.0)
    assert not result["bslow_update_mask"][detach_start:].any()
    assert np.ptp(result["Bslow"][detach_start - 1 :]) == 0.0


def test_two_decompositions_have_independent_outputs(monkeypatch):
    """两路调用不共享事件列表或数组，且各自保持精确分解恒等式。"""
    def simple_pilot(clean, fs, config):
        return {
            "Bpilot": clean.copy(),
            "fast_residual": np.zeros_like(clean),
            "pilot_slope": np.gradient(clean) * fs,
        }

    monkeypatch.setattr(contact_module, "estimate_pilot_trend", simple_pilot)
    monkeypatch.setattr(
        contact_module,
        "detect_transition_candidates",
        lambda clean, *_: (
            [],
            {"residual": np.zeros_like(clean), "scores": np.zeros_like(clean), "scales_s": []},
        ),
    )
    signal2 = np.linspace(10.0, 10.001, 100)
    signal3 = np.linspace(20.0, 20.001, 100)

    result2 = decompose_contact_state_udeco(signal2, 20.0)
    result3 = decompose_contact_state_udeco(signal3, 20.0)
    result2["raw"][0] = -999.0
    result2["transition_candidates"].append({"channel": 2})

    assert result3["raw"][0] != -999.0
    assert result3["transition_candidates"] == []
    for result in (result2, result3):
        reconstructed = result["J"] + result["P"] + result["Bslow"] + result["residual"]
        np.testing.assert_allclose(reconstructed, result["clean"], atol=1e-12)


def test_conditional_bslow_rejects_mismatched_diagnostics():
    """最必要的形状错误应在基线接口处给出明确提示。"""
    bm = np.ones(20)
    state = {
        "contact_mask": np.ones(20, dtype=bool),
        "noncontact_mask": np.zeros(20, dtype=bool),
        "transition_mask": np.zeros(20, dtype=bool),
        "uncertain_mask": np.zeros(20, dtype=bool),
    }
    with pytest.raises(ValueError, match="等长"):
        estimate_conditional_bslow(
            bm,
            state,
            {"dBm_dt": np.zeros(19), "scores": np.zeros(20)},
            20.0,
            ContactStateConfig(),
        )
