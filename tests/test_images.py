"""Image utilities: padding, quadrants, augmentation."""

import os

import numpy as np

from ai_toolset.images import augment_dir, pad_to_square, split_quadrants


def test_pad_to_square_non_square():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    padded, p = pad_to_square(frame)
    assert padded.shape[0] == padded.shape[1] == 200
    assert p > 0


def test_split_quadrants():
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    parts = split_quadrants(img)
    assert len(parts) == 4
    assert parts[0].shape == (40, 40, 3)


def test_augment_dir(tmp_path, synthetic_image):
    src = os.path.join(tmp_path, "src")
    dst = os.path.join(tmp_path, "dst")
    os.makedirs(src)
    from cv2 import imwrite

    imwrite(os.path.join(src, "a.jpg"), synthetic_image)
    count = augment_dir(src, dst, ops=["hflip", "vflip", "rot90", "bright"])
    assert count > 0
    files = os.listdir(dst)
    assert any(f.endswith("_hflip.jpg") for f in files)
