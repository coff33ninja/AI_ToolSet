"""Synthetic object-detection dataset generation.

Generates images of solid-colored rectangles standing in for object classes,
plus keras-retinanet style CSV labels, for smoke-testing a detection pipeline
end to end on the GPU before committing to real labeled data.
"""

import os
import random

import cv2
import numpy as np


def generate_synthetic(classes, out_dir, img_size=320, n_train=40, n_val=12,
                       seed=42, min_boxes=3, max_boxes=8, min_size=24, max_size=56):
    """Generate a synthetic detection dataset.

    classes: dict class_name -> BGR color tuple, e.g. {"npc": (50, 50, 220)}.
    Writes classes.csv, train_labels.csv, val_labels.csv, train/ and val/.
    Returns the output directory path.
    """
    data_dir = out_dir
    os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "val"), exist_ok=True)

    with open(os.path.join(data_dir, "classes.csv"), "w") as f:
        f.writelines(f"{cls},{i}\n" for i, cls in enumerate(classes))

    rng = random.Random(seed)

    def make_image(out_path, labels_out):
        img = np.full((img_size, img_size, 3), (30, 30, 30), dtype=np.uint8)
        boxes = []
        n_boxes = rng.randint(min_boxes, max_boxes)
        attempts = 0
        while len(boxes) < n_boxes and attempts < 200:
            attempts += 1
            w = rng.randint(min_size, max_size)
            h = rng.randint(min_size, max_size)
            x1 = rng.randint(8, img_size - w - 8)
            y1 = rng.randint(8, img_size - h - 8)
            x2 = x1 + w
            y2 = y1 + h
            if any(not (x2 < bx1 or bx2 < x1 or y2 < by1 or by2 < y1) for bx1, by1, bx2, by2, _ in boxes):
                continue
            cls = rng.choice(list(classes.keys()))
            boxes.append((x1, y1, x2, y2, cls))
            cv2.rectangle(img, (x1, y1), (x2, y2), classes[cls], thickness=-1)
        cv2.imwrite(out_path, img)
        for x1, y1, x2, y2, cls in boxes:
            labels_out.append("{},{},{},{},{},{}".format(
                os.path.relpath(out_path, data_dir).replace("\\", "/"),
                x1, y1, x2, y2, cls))

    for split, count, labels_file in (("train", n_train, "train_labels.csv"),
                                      ("val", n_val, "val_labels.csv")):
        labels_out = []
        for i in range(count):
            out_path = os.path.join(data_dir, split, f"img_{i:03d}.png")
            make_image(out_path, labels_out)
        with open(os.path.join(data_dir, labels_file), "w") as f:
            f.write("\n".join(labels_out) + "\n")
        print(f"{split}: {count} images, {len(labels_out)} boxes -> {labels_file}")

    return data_dir
