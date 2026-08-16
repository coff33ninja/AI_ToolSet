"""Synthetic dataset generation and video frame tooling."""

import os

import cv2

from ai_toolset.dataset import generate_synthetic
from ai_toolset.video import extract_frames, frames_to_video


def test_generate_synthetic(tmp_path):
    out_dir = os.path.join(tmp_path, "ds")
    generate_synthetic(
        classes={"bottle": (50, 50, 220), "cup": (220, 50, 50)},
        out_dir=out_dir,
        img_size=128,
        n_train=2,
        n_val=1,
    )
    assert os.path.isdir(os.path.join(out_dir, "train"))
    assert os.path.isdir(os.path.join(out_dir, "val"))
    assert os.path.isfile(os.path.join(out_dir, "classes.csv"))
    assert os.path.isfile(os.path.join(out_dir, "train_labels.csv"))
    assert os.path.isfile(os.path.join(out_dir, "val_labels.csv"))
    assert len(os.listdir(os.path.join(out_dir, "train"))) == 2


def test_extract_frames(tmp_path, sample_video_path):
    out_dir = os.path.join(tmp_path, "frames")
    written = extract_frames(sample_video_path, out_dir, mode="interval", min_skip=5)
    assert written >= 2
    assert len(os.listdir(out_dir)) >= 2


def test_frames_to_video_roundtrip(tmp_path, synthetic_image):
    frames_dir = os.path.join(tmp_path, "frames_in")
    os.makedirs(frames_dir)
    for i in range(5):
        cv2.imwrite(os.path.join(frames_dir, f"f{i:04d}.jpg"), synthetic_image)
    out = os.path.join(tmp_path, "out.mp4")
    assert frames_to_video(frames_dir, out, fps=10) is not None
    cap = cv2.VideoCapture(out)
    assert cap.isOpened()
    frames_read = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frames_read == 5
