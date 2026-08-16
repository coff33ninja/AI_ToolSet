"""CUDA/GPU detection, TensorFlow GPU setup, and GPU selection."""

import os
import shutil
import subprocess

TF_CUDA_MATRIX = {
    "2.4": {"cuda": "11.0", "cudnn": "8.0.5.39", "driver_min": 451.82},
    "2.5": {"cuda": "11.2", "cudnn": "8.1.0.77", "driver_min": 460.89},
    "2.6": {"cuda": "11.2", "cudnn": "8.1.0.77", "driver_min": 460.89},
    "2.7": {"cuda": "11.2", "cudnn": "8.1.0.77", "driver_min": 460.89},
    "2.8": {"cuda": "11.2", "cudnn": "8.1.0.77", "driver_min": 460.89},
    "2.9": {"cuda": "11.2", "cudnn": "8.1.0.77", "driver_min": 460.89},
    "2.10": {"cuda": "11.2", "cudnn": "8.1.0.77", "driver_min": 460.89},
}


def detect_gpus():
    """Return a list of {index, name, driver, vram} dicts from nvidia-smi.

    Empty list when nvidia-smi is missing, errors, or finds no GPUs.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    result = []
    for index, line in enumerate(out.stdout.splitlines()):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        result.append({"index": index, "name": parts[1],
                       "driver": float(parts[0]), "vram": parts[2]})
    return result


def detect_gpu():
    """Return the first GPU as {index, name, driver, vram}, or None."""
    gpus = detect_gpus()
    return gpus[0] if gpus else None


def matrix_entry(tf_version):
    """Return the {cuda, cudnn, driver_min} mapping for a TensorFlow version."""
    entry = TF_CUDA_MATRIX.get(tf_version)
    if entry is None:
        raise KeyError(
            "No verified CUDA/cuDNN mapping for TensorFlow {}. Supported: {}".format(
                tf_version, ", ".join(sorted(TF_CUDA_MATRIX))
            )
        )
    return entry


def set_visible_gpus(gpus=None):
    """Restrict GPU use to the given physical GPU indices via CUDA_VISIBLE_DEVICES.

    Call BEFORE importing tensorflow or torch (both honor the env var at
    import time). gpus None/'all'/[] = all GPUs visible. Returns the chosen
    index list, or None for all.
    """
    if gpus in (None, "all") or gpus == []:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return None
    indices = [int(g) for g in gpus]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in indices)
    return indices


def configure_tf_gpus(gpus=None):
    """Restrict TensorFlow to chosen GPU indices at runtime (call after import).

    Enables memory growth on each visible device. gpus None/'all'/[] = all
    physical GPUs. Returns the number of visible GPU devices.
    """
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        return 0
    if gpus not in (None, "all") and gpus != []:
        indices = [int(g) for g in gpus]
        devices = [devices[i] for i in indices if 0 <= i < len(devices)]
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.set_visible_devices(devices, "GPU")
    return len(devices)


def configure_memory_growth():
    """Let TensorFlow grow GPU memory on demand for all devices."""
    return configure_tf_gpus()


def verify_tf_gpu():
    """Return tf.config.list_physical_devices("GPU") result after setup."""
    import tensorflow as tf

    return tf.config.list_physical_devices("GPU")


def verify_torch_gpu():
    """Return a list of CUDA device names visible to PyTorch (empty if none)."""
    import torch

    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
