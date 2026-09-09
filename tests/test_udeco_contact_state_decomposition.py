"""接触状态分解的阶段一回归测试。"""

import numpy as np
import pytest

from udeco_contact_state_decomposition import (
    ContactStateConfig,
    detect_transition_candidates,
    estimate_pilot_trend,
    hampel_filter,
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
