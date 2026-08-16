"""Image helpers: square padding, quadrant splitting, label conversion."""

import csv
import glob
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def pad_to_square(frame, color=(0, 0, 0)):
    """Pad a BGR frame to square, returning (padded, padding_amount)."""
    h, w = frame.shape[:2]
    if h == w:
        return frame, 0
    if h < w:
        p = (w - h) // 2
        padded = cv2.copyMakeBorder(frame, p, w - h - p, 0, 0, cv2.BORDER_CONSTANT, value=color)
        return padded, p
    p = (h - w) // 2
    padded = cv2.copyMakeBorder(frame, 0, 0, p, h - w - p, cv2.BORDER_CONSTANT, value=color)
    return padded, p


def pad_images_in_dir(in_dir, out_dir=None, ext="jpg", color=(0, 0, 0)):
    """Pad every image in a directory to square, writing to out_dir."""
    if out_dir is None:
        out_dir = in_dir
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for path in glob.glob(os.path.join(in_dir, f"*.{ext}")):
        img = cv2.imread(path)
        if img is None:
            continue
        padded, _ = pad_to_square(img, color=color)
        out = os.path.join(out_dir, os.path.basename(path))
        cv2.imwrite(out, padded)
        count += 1
    return count


def split_quadrants(image):
    """Split an image into 4 quadrants: (top_left, top_right, bot_left, bot_right)."""
    h, w = image.shape[:2]
    mid_h, mid_w = h // 2, w // 2
    return (
        image[0:mid_h, 0:mid_w],
        image[0:mid_h, mid_w:w],
        image[mid_h:h, 0:mid_w],
        image[mid_h:h, mid_w:w],
    )


def _read_labels(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for line in csv.reader(f):
            if len(line) < 8 or line[0] == "filename":
                continue
            rows.append({
                "filename": line[0],
                "width": int(line[1]),
                "height": int(line[2]),
                "class": line[3],
                "xmin": int(line[4]),
                "ymin": int(line[5]),
                "xmax": int(line[6]),
                "ymax": int(line[7]),
            })
    return rows


_QUAD_BOUNDS = [
    (0, 0),  # top_left    x[0:1], y[0:1]
    (1, 0),  # top_right   x[1:2], y[0:1]
    (0, 1),  # bot_left    x[0:1], y[1:2]
    (1, 1),  # bot_right   x[1:2], y[1:2]
]


def split_image_dataset(image_dir, labels_csv, out_dir, min_area_ratio=0.25):
    """Quadrant-split images and relabel boxes.

    Labels CSV columns: filename,width,height,class,xmin,ymin,xmax,ymax.
    Images must live in image_dir; label filenames are matched by basename.
    Returns the number of output label rows written to out_dir/labels.csv.
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = _read_labels(labels_csv)
    rows = []
    for img_path in glob.glob(os.path.join(image_dir, "*")):
        if not os.path.isfile(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        name = os.path.basename(img_path)
        h, w = img.shape[:2]
        seg_w = [0, w // 2, w]
        seg_h = [0, h // 2, h]
        quads = split_quadrants(img)
        qnames = ["top_left_{}", "top_right_{}", "bot_left_{}", "bot_right_{}"]
        for qi, crop in enumerate(quads):
            out_name = qnames[qi].format(name)
            cv2.imwrite(os.path.join(out_dir, out_name), crop)
            xi, yi = _QUAD_BOUNDS[qi]
            for label in labels:
                if os.path.basename(label["filename"]) != name:
                    continue
                xmin, ymin = label["xmin"], label["ymin"]
                xmax, ymax = label["xmax"], label["ymax"]
                if not (xmin < seg_w[xi + 1] and xmax > seg_w[xi]
                        and ymin < seg_h[yi + 1] and ymax > seg_h[yi]):
                    continue
                nx1 = max(xmin, seg_w[xi]) - seg_w[xi]
                nx2 = min(xmax, seg_w[xi + 1]) - seg_w[xi]
                ny1 = max(ymin, seg_h[yi]) - seg_h[yi]
                ny2 = min(ymax, seg_h[yi + 1]) - seg_h[yi]
                if (nx2 - nx1) * (ny2 - ny1) < min_area_ratio * (xmax - xmin) * (ymax - ymin):
                    continue
                rows.append((out_name, nx1, ny1, nx2, ny2, label["class"]))
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "xmin", "ymin", "xmax", "ymax", "class"])
        writer.writerows(rows)
    return len(rows)


def xml_to_csv(xml_dir, out_csv):
    """Convert labelImg/PascalVOC XML files into a keras-retinanet CSV.

    Output format: path,xmin,ymin,xmax,ymax,class_name
    """
    rows = []
    for xml_file in glob.glob(os.path.join(xml_dir, "*.xml")):
        root = ET.parse(xml_file).getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name")
            bbox = obj.find("bndbox")
            if bbox is None or name is None:
                continue
            rows.append((
                root.findtext("filename"),
                int(bbox.findtext("xmin")),
                int(bbox.findtext("ymin")),
                int(bbox.findtext("xmax")),
                int(bbox.findtext("ymax")),
                name,
            ))
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "xmin", "ymin", "xmax", "ymax", "class"])
        writer.writerows(rows)
    return len(rows)


_AUGMENT_OPS = {
    "hflip": lambda img: cv2.flip(img, 1),
    "vflip": lambda img: cv2.flip(img, 0),
    "rot90": lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
    "rot180": lambda img: cv2.rotate(img, cv2.ROTATE_180),
    "rot270": lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    "blur": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
}


def _brightness_contrast(img):
    alpha = float(np.random.uniform(0.7, 1.3))  # contrast
    beta = int(np.random.uniform(-40, 40))       # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def _hue_shift(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(np.random.uniform(-20, 20))) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


_AUGMENT_RANDOM_OPS = {
    "bright": _brightness_contrast,
    "hue": _hue_shift,
}


def augment_dir(in_dir, out_dir=None, ext="jpg", ops=None):
    """Expand a dataset with deterministic + random augmentations.

    Each input image produces one output per op, written as <name>_<op>.<ext>.
    Default ops: hflip, vflip, rot90, rot180, rot270, blur, bright, hue.
    Returns the number of images written.
    """
    if out_dir is None:
        out_dir = in_dir
    os.makedirs(out_dir, exist_ok=True)
    if ops is None:
        ops = list(_AUGMENT_OPS) + list(_AUGMENT_RANDOM_OPS)
    count = 0
    for path in sorted(glob.glob(os.path.join(in_dir, f"*.{ext}"))):
        img = cv2.imread(path)
        if img is None:
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        for op in ops:
            fn = _AUGMENT_OPS.get(op) or _AUGMENT_RANDOM_OPS.get(op)
            if fn is None:
                raise ValueError(f"Unknown augmentation op: {op}")
            out = os.path.join(out_dir, f"{stem}_{op}.{ext}")
            cv2.imwrite(out, fn(img))
            count += 1
    return count
