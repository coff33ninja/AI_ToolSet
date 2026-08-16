"""Select which GPU(s) to use, then verify TensorFlow/PyTorch visibility.

Both GPUs (GTX 1060 + GTX 1070) or just one can be selected. The selection
must happen BEFORE tensorflow/torch are imported, which is why set_visible_gpus
is called first in this file.

Usage:
    uv run python examples/select_gpu.py --gpus 0,1     # both GPUs
    uv run python examples/select_gpu.py --gpus 1       # only the second
    uv run python examples/select_gpu.py                 # interactive prompt
"""

import argparse

from ai_toolset import cuda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus", default=None, help="comma-separated physical GPU indices (e.g. 0,1) or 'all'"
    )
    parser.add_argument("--framework", choices=["tf", "torch", "auto"], default="auto")
    args = parser.parse_args()

    detected = cuda.detect_gpus()
    if not detected:
        print("No NVIDIA GPU detected via nvidia-smi.")
        raise SystemExit(1)
    for g in detected:
        print(f"[{g['index']}] {g['name']}  {g['vram']}  driver {g['driver']}")

    chosen = None
    if args.gpus and args.gpus != "all":
        chosen = [int(x) for x in args.gpus.split(",") if x.strip()]
    elif not args.gpus and len(detected) > 1:
        reply = input("Which GPU(s)? comma-separated indices or 'all': ").strip()
        if reply and reply.lower() != "all":
            chosen = [int(x) for x in reply.split(",") if x.strip()]

    cuda.set_visible_gpus(chosen)
    print("Visible GPUs:", chosen if chosen is not None else "all")

    if args.framework in ("torch", "auto"):
        try:
            names = cuda.verify_torch_gpu()
            print("PyTorch sees:", names if names else "no CUDA device")
        except ImportError:
            print("PyTorch not installed (uv sync --extra voice).")

    if args.framework in ("tf", "auto"):
        try:
            devices = cuda.verify_tf_gpu()
            print("TensorFlow sees:", devices if devices else "no GPU device")
        except ImportError:
            print("TensorFlow not installed (uv sync).")


if __name__ == "__main__":
    main()
