"""独立边沿检测的新算法回归测试。"""

import numpy as np

from udeco_edge_detection import CascadeStateConfig, detect_multiscale_edges


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
