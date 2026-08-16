"""YOLO detection (ultralytics) on images, video, and live screen streams.

All models are downloaded into the local Hugging Face / ultralytics cache by
ultralytics itself - nothing is installed system-wide. Ultralytics uses the
cu118 torch from the voice extra; GPU selection happens through
CUDA_VISIBLE_DEVICES (ai_toolset.cuda.set_visible_gpus) before the torch
import, exactly like the speech backends.
"""

import os
import time

import cv2

from ai_toolset.cuda import set_visible_gpus

DEFAULT_WEIGHTS = "yolov8n.pt"
DEFAULT_CONF = 0.25


def _load(weights=DEFAULT_WEIGHTS, gpus=None):
    """Import ultralytics and load the model, honoring GPU selection."""
    set_visible_gpus(gpus)
    from ultralytics import YOLO

    return YOLO(weights)


def _to_detections(result, names):
    """Flatten one ultralytics Result into [{label, conf, box}] dicts."""
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    for x1, y1, x2, y2, c, p in zip(xyxy[:, 0], xyxy[:, 1],
                                    xyxy[:, 2], xyxy[:, 3], cls, conf):
        detections.append({
            "label": names.get(c, str(c)),
            "conf": float(p),
            "box": [int(x1), int(y1), int(x2), int(y2)],
        })
    return detections


def detect_image(path, weights=DEFAULT_WEIGHTS, conf=DEFAULT_CONF,
                 device="auto", gpus=None):
    """Run YOLO on an image file. Returns (detections, annotated-image-path).

    When device is None/'auto', the model is left on ultralytics' default
    (cuda if available). With gpus= set, only those physical GPUs are visible.
    """
    model = _load(weights, gpus=gpus)
    if device == "auto":
        device = None
    result = model.predict(path, conf=conf, device=device, verbose=False)[0]
    return _to_detections(result, model.names), result


def detect_frame(frame, weights=DEFAULT_WEIGHTS, conf=DEFAULT_CONF,
                 gpus=None, model=None):
    """Run YOLO on a BGR numpy frame.

    Pass a previously loaded `model` to reuse it across frames (model loading
    is the expensive part). Returns the detections list.
    """
    if model is None:
        model = _load(weights, gpus=gpus)
    result = model.predict(frame, conf=conf, verbose=False)[0]
    return _to_detections(result, model.names)


def draw_detections(frame, detections, label_color=(0, 255, 0)):
    """Draw detection boxes + labels onto a BGR frame (in place)."""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)
        text = f"{det['label']} {det['conf']:.2f}"
        cv2.putText(frame, text, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2)
    return frame


def detect_stream(region=None, weights=DEFAULT_WEIGHTS, conf=DEFAULT_CONF,
                  gpus=None, fps_window=30):
    """Yield (frame, detections, fps) for a live screen region.

    Loads the model once, then captures frames via ai_toolset.screen and runs
    detection on each, reporting a rolling FPS. Returns nothing - consume the
    generator (e.g. from the CLI preview window).
    """
    from ai_toolset.screen import stream_frames

    model = _load(weights, gpus=gpus)
    timings = []
    for frame in stream_frames(region):
        start = time.perf_counter()
        detections = detect_frame(frame, gpus=gpus, model=model, conf=conf)
        timings.append(time.perf_counter() - start)
        if len(timings) > fps_window:
            timings.pop(0)
        fps = len(timings) / sum(timings) if timings else 0.0
        yield frame, detections, fps


def annotate(path, out_path=None, weights=DEFAULT_WEIGHTS, conf=DEFAULT_CONF,
             gpus=None):
    """Detect on an image and write an annotated copy. Returns out_path."""
    detections, _ = detect_image(path, weights=weights, conf=conf, gpus=gpus)
    frame = cv2.imread(path)
    if frame is None:
        raise OSError(f"Could not read image: {path}")
    draw_detections(frame, detections)
    out_path = out_path or os.path.splitext(path)[0] + "_detected.jpg"
    cv2.imwrite(out_path, frame)
    return out_path
