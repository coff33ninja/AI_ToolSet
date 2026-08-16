"""Run YOLO detection on an image and write an annotated copy (ai-toolset).

Usage:
  uv run python examples/detect_image.py frame.png out.png --gpus 0
"""

import argparse

from ai_toolset import detect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="image to run YOLO on")
    parser.add_argument("out", help="path to write the annotated image")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--gpus", help="comma-separated GPU indices, e.g. 0,1")
    args = parser.parse_args()

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    out = detect.annotate(args.image, args.out, weights=args.weights,
                          conf=args.conf, gpus=gpus)
    print(f"annotated -> {out}")


if __name__ == "__main__":
    main()
