"""MediaPipe overlay (skipped without the mediapipe extra) and YOLO drawing."""

import os

import numpy as np
import pytest

from ai_toolset import detect
from ai_toolset.mp import available, overlay


def test_draw_detections_synthetic(synthetic_image):
    detections = [
        {"label": "bottle", "conf": 0.9, "box": [10, 10, 90, 90]},
        {"label": "cup", "conf": 0.6, "box": [100, 100, 180, 180]},
    ]
    out = detect.draw_detections(synthetic_image, detections)
    assert out.shape == synthetic_image.shape


@pytest.mark.slow
def test_detect_frame_cached_weights(synthetic_image):
    if not os.path.isfile(detect.DEFAULT_WEIGHTS):
        pytest.skip(f"{detect.DEFAULT_WEIGHTS} not downloaded")
    detections, _ = detect.detect_frame(synthetic_image)
    assert isinstance(detections, list)
    for det in detections:
        assert {"label", "conf", "box"}.issubset(det)


@pytest.mark.skipif(not available(), reason="mediapipe extra not installed")
def test_overlay_pose(synthetic_image):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = overlay(frame, solution="pose", static=True)
    assert out.shape == frame.shape
