"""独立边沿检测的新算法回归测试。"""

import numpy as np
import pytest

from udeco_edge_detection import (
    CascadeStateConfig,
    classify_edge_candidates,
    decompose_cascade_udeco,
    detect_multiscale_edges,
    infer_contact_mode,
    measure_edge_morphology,
    median_clean,
)


def test_derivative_channel_finds_one_sample_up_and_down_steps():
    signal = np.full(240, 0.02)
    signal[80:160] += 0.01

    candidates, diagnostics = detect_multiscale_edges(
        signal, np.zeros_like(signal), 100.0, CascadeStateConfig()
    )

    for center in (80, 160):
        nearby = [
            item for item in candidates
            if abs(item["center_index"] - center) <= 1
            and "derivative" in item["detection_channels"]
        ]
        assert nearby, (center, candidates)
    assert diagnostics["derivative_score"].shape == signal.shape
    assert all("candidate_type" not in item for item in candidates)


def test_nearby_same_direction_steps_remain_separate_candidates():
    signal = np.full(240, 0.02)
    signal[80:] += 0.008
    signal[86:] += 0.008

    candidates, _ = detect_multiscale_edges(
        signal, np.zeros_like(signal), 100.0, CascadeStateConfig()
    )

    derivative_centers = [
        item["center_index"] for item in candidates
        if "derivative" in item["detection_channels"]
    ]
    assert any(abs(center - 80) <= 1 for center in derivative_centers)
    assert any(abs(center - 86) <= 1 for center in derivative_centers)


def _ramp_signal(start, end, delta, *, drift=0.0, curve=1.0, overshoot=0.0):
    signal = 0.02 + drift * np.arange(240)
    progress = np.clip((np.arange(240) - start) / (end - start), 0.0, 1.0)
    signal = signal + delta * progress**curve
    if overshoot:
        signal[end + 1] += overshoot
    return signal


@pytest.mark.parametrize("start,end,delta,tolerance", [
    (90, 91, 0.01, 1),
    (90, 92, -0.01, 1),
    (90, 110, -0.01, 2),
])
def test_fitted_boundaries_track_fast_and_slow_ramps(start, end, delta, tolerance):
    signal = _ramp_signal(start, end, delta)
    candidate = {
        "center_index": (start + end) // 2,
        "best_scale_s": 0.04,
        "edge_score": 8.0,
        "noise_scale": 1e-5,
    }

    fitted = measure_edge_morphology(
        signal, candidate, 100.0, CascadeStateConfig(), 1e-5
    )
    classified = classify_edge_candidates([fitted], CascadeStateConfig())[0]

    assert abs(fitted["start_index"] - start) <= tolerance
    assert abs(fitted["end_index"] - end) <= tolerance
    assert classified["candidate_type"] == ("detach" if delta > 0 else "attach")
    assert fitted["edge_snr"] >= 3
    assert fitted["fit_improvement"] >= 0.15
    assert fitted["edge_width_points"] == fitted["end_index"] - fitted["start_index"]


@pytest.mark.parametrize("kwargs", [
    {"drift": 1e-6},
    {"curve": 1.4},
    {"overshoot": -0.0015},
])
def test_fitted_slow_drop_handles_drift_curvature_and_overshoot(kwargs):
    signal = _ramp_signal(90, 110, -0.01, **kwargs)
    candidate = {
        "center_index": 100,
        "best_scale_s": 0.08,
        "edge_score": -8.0,
        "noise_scale": 1e-5,
    }
    fitted = measure_edge_morphology(
        signal, candidate, 100.0, CascadeStateConfig(), 1e-5
    )
    classified = classify_edge_candidates([fitted], CascadeStateConfig())[0]

    assert abs(fitted["start_index"] - 90) <= 2
    assert abs(fitted["end_index"] - 110) <= 2
    assert classified["candidate_type"] == "attach"
    assert fitted["left_mad"] >= 0 and fitted["right_mad"] >= 0
    assert fitted["fit_residual"] >= 0


def test_local_nonmonotonic_rebound_does_not_reject_curved_attach():
    signal = _ramp_signal(90, 110, -0.01, curve=1.3)
    signal[100] += 0.001
    signal[101] += 0.0006
    candidate = {
        "center_index": 100, "best_scale_s": 0.08,
        "edge_score": -8.0, "noise_scale": 1e-5,
    }
    fitted = measure_edge_morphology(
        signal, candidate, 100.0, CascadeStateConfig(), 1e-5
    )
    classified = classify_edge_candidates([fitted], CascadeStateConfig())[0]

    assert abs(fitted["start_index"] - 90) <= 2
    assert abs(fitted["end_index"] - 110) <= 2
    assert fitted["monotonicity"] < 1.0
    assert classified["candidate_type"] == "attach"


def test_short_plateau_and_dense_edges_remain_independent():
    signal = np.full(200, 0.02)
    signal[80:86] += 0.008
    signal[86:] += 0.016
    clean = median_clean(signal, CascadeStateConfig())
    result = decompose_cascade_udeco(clean, 100.0)
    nearby = [
        item for item in result["J_events"]
        if 77 <= item["center_index"] <= 88
    ]
    assert [item["event_type"] for item in nearby] == ["detach", "detach"]
    assert abs(nearby[0]["start_index"] - 79) <= 1
    assert abs(nearby[1]["start_index"] - 85) <= 1


def test_missed_opposite_edge_does_not_suppress_next_detach():
    candidates = [
        {"center_index": 80, "start_index": 79, "end_index": 80,
         "candidate_type": "detach", "confidence": 0.9},
        {"center_index": 120, "candidate_type": "ambiguous", "confidence": 0.1},
        {"center_index": 160, "start_index": 159, "end_index": 160,
         "candidate_type": "detach", "confidence": 0.9},
    ]
    state = infer_contact_mode(candidates, 200, 100.0, CascadeStateConfig())

    assert [item["center_index"] for item in state["J_events"]] == [80, 160]
    assert state["noncontact_mask"][81:159].all()
    assert state["noncontact_mask"][161:].all()
    assert state["edge_candidates"][2]["accepted"]


def test_monotonicity_is_soft_and_absolute_level_is_irrelevant():
    up = {
        "delta": 0.005, "edge_snr": 8.0, "fit_improvement": 0.7,
        "monotonicity": 0.4, "left_level": -0.2, "right_level": -0.195,
    }
    down = {**up, "delta": -0.005, "left_level": 0.5, "right_level": 0.495}
    classified = classify_edge_candidates([up, down], CascadeStateConfig())
    assert [item["candidate_type"] for item in classified] == ["detach", "attach"]
    assert all(item["confidence"] > 0 for item in classified)


def test_gradual_pressure_without_local_plateau_step_stays_contact():
    fs = 100.0
    time = np.arange(int(30 * fs)) / fs
    signal = 0.02 + 2e-5 * time
    loading = (time >= 8.0) & (time < 13.0)
    hold = (time >= 13.0) & (time < 18.0)
    relaxing = (time >= 18.0) & (time < 24.0)
    signal[loading] += 0.004 * (time[loading] - 8.0) / 5.0
    signal[hold] += 0.004
    signal[relaxing] += 0.004 * (1.0 - (time[relaxing] - 18.0) / 6.0)

    result = decompose_cascade_udeco(signal, fs)

    assert not result["J_events"]
    assert not result["noncontact_mask"].any()
    np.testing.assert_allclose(
        result["clean"],
        result["J"] + result["P"] + result["Bslow"] + result["residual"],
    )
