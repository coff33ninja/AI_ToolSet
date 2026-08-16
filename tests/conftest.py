import os

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_image():
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (200, 200), (0, 0, 255), -1)
    return img


@pytest.fixture
def synthetic_image_path(tmp_path, synthetic_image):
    path = os.path.join(tmp_path, "test.jpg")
    cv2.imwrite(path, synthetic_image)
    return path


@pytest.fixture
def tone_wav_path(tmp_path):
    path = os.path.join(tmp_path, "tone.wav")
    import wave

    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5))
    data = (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((data * 32767).astype("<i2").tobytes())
    return path


@pytest.fixture
def sample_video_path(tmp_path):
    path = os.path.join(tmp_path, "clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 20.0, (128, 128))
    assert writer.isOpened(), "could not open video writer"
    for _ in range(20):
        writer.write(np.zeros((128, 128, 3), dtype=np.uint8))
    writer.release()
    return path
