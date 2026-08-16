"""YOLO training wrapper (ultralytics).

Thin wrapper over ultralytics' YOLO.train() so the CLI stays consistent with
the rest of the toolkit. Dataset layout is ultralytics' standard:

    dataset/
      images/train, images/val
      labels/train, labels/val
      data.yaml   (path, nc, names, train/val keys)

The dataset-prep commands (make-synthetic, split-images, xml-to-csv,
capture-loop, extract-frames) produce the images + labels; train.py only needs
the data.yaml. Weights download into the local ultralytics cache on first use.
"""

import os


def train_yolo(
    data_yaml,
    model="yolov8n.pt",
    epochs=100,
    imgsz=640,
    batch=16,
    gpus=None,
    project="runs",
    name="detect",
    exist_ok=False,
    patience=50,
):
    """Train a YOLO model on a dataset. Returns the training results object.

    gpus= restricts which physical GPUs are visible (via CUDA_VISIBLE_DEVICES
    before the torch import) - pass "0,1" to train on both cards. project/name
    follow ultralytics conventions: weights land in <project>/<name>/weights/.
    """
    from ai_toolset.cuda import set_visible_gpus

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"No dataset config: {data_yaml}")
    set_visible_gpus(gpus)

    from ultralytics import YOLO

    model_obj = YOLO(model)
    return model_obj.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="0",
        project=project,
        name=name,
        exist_ok=exist_ok,
        patience=patience,
    )


def best_weights(project="runs", name="detect"):
    """Return the path to the best trained weights, or None."""
    candidate = os.path.join(project, name, "weights", "best.pt")
    return candidate if os.path.exists(candidate) else None
