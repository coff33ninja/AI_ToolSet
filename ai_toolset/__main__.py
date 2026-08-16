"""CLI for the ai-toolset. Run: uv run python -m ai_toolset --help"""

import argparse
import json
import os
import sys

import cv2


def cmd_verify_cuda(args):
    from ai_toolset import cuda

    gpus = cuda.detect_gpus()
    if not gpus:
        print("No NVIDIA GPU detected via nvidia-smi.")
        return 1
    for gpu in gpus:
        print(f"GPU [{gpu['index']}]: {gpu['name']} | driver {gpu['driver']} | VRAM {gpu['vram']}")
    if args.tf:
        entry = cuda.matrix_entry(args.tf)
        ok = all(gpu["driver"] >= entry["driver_min"] for gpu in gpus)
        print(f"TF {args.tf} needs CUDA {entry['cuda']} + cuDNN {entry['cudnn']}, "
              f"driver >= {entry['driver_min']} -> {'OK' if ok else 'UPDATE DRIVER'}")
        print("GPU selection: python -m ai_toolset select-gpus --gpus 0,1  (or 'all')")
        return 0 if ok else 2
    return 0


def cmd_select_gpus(args):
    from ai_toolset import cuda

    gpus = cuda.detect_gpus()
    if not gpus:
        print("No NVIDIA GPUs detected via nvidia-smi.")
        return 1
    for gpu in gpus:
        print(f"  [{gpu['index']}] {gpu['name']}  {gpu['vram']}  driver {gpu['driver']}")

    chosen = None
    if args.gpus:
        chosen = [int(x) for x in args.gpus.split(",") if x.strip()]
        for i in chosen:
            if not (0 <= i < len(gpus)):
                print(f"Bad GPU index {i}; valid: 0..{len(gpus) - 1}")
                return 2
    elif len(gpus) > 1:
        reply = input("Which GPU(s)? comma-separated indices or 'all': ").strip().lower()
        if reply and reply != "all":
            chosen = [int(x) for x in reply.split(",") if x.strip()]

    cuda.set_visible_gpus(chosen)
    label = "all GPUs" if chosen is None else f"GPU {chosen}"
    print(f"Selected {label}. CUDA_VISIBLE_DEVICES is set in this process; "
          "set it the same way before importing tensorflow/torch in your scripts.")
    return 0


def cmd_select_region(args):
    from ai_toolset.screen import select_region

    region = select_region()
    if region is None:
        print("Cancelled.")
        return 1
    print(json.dumps(region))
    return 0


def cmd_capture_loop(args):
    from ai_toolset import screen

    region = json.loads(args.region) if args.region else None
    os.makedirs(args.out_dir, exist_ok=True)
    index = 0
    for frame in screen.stream_frames(region):
        if args.square:
            frame, _ = screen.pad_to_square(frame)
        cv2.imshow("capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.imwrite(os.path.join(args.out_dir, f"{index:05d}.jpg"), frame)
            index += 1
            print(f"Saved frame {index}")
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    return 0


def cmd_extract_frames(args):
    from ai_toolset.video import extract_frames

    saved = extract_frames(args.video, args.out_dir, mode=args.mode,
                           count=args.count, min_skip=args.min_skip,
                           max_skip=args.max_skip, start_index=args.start_index)
    print(f"Extracted {saved} frames to {args.out_dir}")
    return 0


def cmd_frames_to_video(args):
    from ai_toolset.video import frames_to_video

    written = frames_to_video(args.frames_dir, args.out_path, fps=args.fps,
                              ext=args.ext, size=args.size)
    print(f"Wrote {written} frames to {args.out_path}")
    return 0


def cmd_make_synthetic(args):
    from ai_toolset.dataset import generate_synthetic

    classes = {name: tuple(int(c) for c in color.split(",")) for name, color in
               (pair.split(":") for pair in args.classes)}
    data_dir = generate_synthetic(classes, args.out_dir, img_size=args.img_size,
                                  n_train=args.train, n_val=args.val, seed=args.seed)
    print(f"Dataset written to {data_dir}")
    return 0


def cmd_pad_images(args):
    from ai_toolset.images import pad_images_in_dir

    count = pad_images_in_dir(args.in_dir, args.out_dir, ext=args.ext)
    print(f"Padded {count} images")
    return 0


def cmd_split_images(args):
    from ai_toolset.images import split_image_dataset

    rows = split_image_dataset(args.image_dir, args.labels_csv, args.out_dir,
                               min_area_ratio=args.min_area_ratio)
    print(f"Wrote {rows} label rows to {args.out_dir}")
    return 0


def cmd_xml_to_csv(args):
    from ai_toolset.images import xml_to_csv

    rows = xml_to_csv(args.xml_dir, args.out_csv)
    print(f"Wrote {rows} rows to {args.out_csv}")
    return 0


def cmd_audio_probe(args):
    from ai_toolset.audio import probe_dir

    entries = probe_dir(args.src_dir)
    for e in entries:
        print(f"{e['duration_sec']:9.2f}s  {e['name']}")
    total = sum(e["duration_sec"] for e in entries)
    print(f"{len(entries)} files, {total:.1f}s total")
    return 0


def cmd_audio_resample(args):
    from ai_toolset.audio import resample_dir

    count = resample_dir(args.src_dir, args.out_dir, sr=args.sr, mono=args.mono)
    print(f"Wrote {count} WAV files to {args.out_dir} (sr={args.sr}, mono={args.mono})")
    return 0


def cmd_audio_silence(args):
    from ai_toolset.audio import split_on_silence

    written = split_on_silence(args.audio_file, args.out_dir, sr=args.sr,
                               max_sec=args.max_sec, min_silence=args.min_silence,
                               top_db=args.top_db)
    print(f"Wrote {len(written)} segments to {args.out_dir}")
    return 0


def cmd_audio_rvc(args):
    from ai_toolset.audio import make_rvc_dataset

    count = make_rvc_dataset(args.src_dir, args.out_dir, speaker_id=args.speaker,
                             max_sec=args.max_sec, sr=args.sr, top_db=args.top_db)
    print(f"Wrote {count} segments to {args.out_dir} (speaker {args.speaker})")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m ai_toolset",
                                     description="Reusable AI toolkit CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("verify-cuda", help="detect all GPUs/driver and CUDA requirements")
    p.add_argument("--tf", default="2.10", help="TensorFlow version to check (default 2.10)")
    p.set_defaults(func=cmd_verify_cuda)

    p = sub.add_parser("select-gpus", help="choose which GPU(s) to use (both or user-selected)")
    p.add_argument("--gpus", help="comma-separated physical GPU indices, e.g. '0,1' or 'all'")
    p.set_defaults(func=cmd_select_gpus)

    p = sub.add_parser("select-region", help="interactively select a screen region")
    p.set_defaults(func=cmd_select_region)

    p = sub.add_parser("capture-loop", help="capture screen frames; s=save, q=quit")
    p.add_argument("--region", help="JSON region dict {top,left,width,height}")
    p.add_argument("--out-dir", default="captured_frames", help="output directory")
    p.add_argument("--square", action="store_true", help="pad frames to square")
    p.set_defaults(func=cmd_capture_loop)

    p = sub.add_parser("extract-frames", help="extract frames from a video")
    p.add_argument("video")
    p.add_argument("out_dir")
    p.add_argument("--mode", choices=["random", "interval"], default="random")
    p.add_argument("--count", type=int, default=0, help="max frames (0=all)")
    p.add_argument("--min-skip", type=int, default=100)
    p.add_argument("--max-skip", type=int, default=2500)
    p.add_argument("--start-index", type=int, default=0)
    p.set_defaults(func=cmd_extract_frames)

    p = sub.add_parser("frames-to-video", help="stitch frames into a video")
    p.add_argument("frames_dir")
    p.add_argument("out_path")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--ext", default="jpg")
    p.add_argument("--size", nargs=2, type=int, help="output WxH")
    p.set_defaults(func=cmd_frames_to_video)

    p = sub.add_parser("make-synthetic", help="generate a synthetic detection dataset")
    p.add_argument("--classes", required=True,
                   help="name:BGR,R,G,B;name:BGR,R,G,B  (e.g. npc:50,50,220)")
    p.add_argument("--out-dir", default="synthetic_data")
    p.add_argument("--img-size", type=int, default=320)
    p.add_argument("--train", type=int, default=40)
    p.add_argument("--val", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_make_synthetic)

    p = sub.add_parser("pad-images", help="pad all images in a dir to square")
    p.add_argument("in_dir")
    p.add_argument("out_dir", nargs="?")
    p.add_argument("--ext", default="jpg")
    p.set_defaults(func=cmd_pad_images)

    p = sub.add_parser("split-images", help="quadrant-split images and relabel")
    p.add_argument("image_dir")
    p.add_argument("labels_csv")
    p.add_argument("out_dir")
    p.add_argument("--min-area-ratio", type=float, default=0.25)
    p.set_defaults(func=cmd_split_images)

    p = sub.add_parser("xml-to-csv", help="convert PascalVOC XML labels to retinanet CSV")
    p.add_argument("xml_dir")
    p.add_argument("out_csv")
    p.set_defaults(func=cmd_xml_to_csv)

    p = sub.add_parser("audio-probe", help="list audio files with durations")
    p.add_argument("src_dir")
    p.set_defaults(func=cmd_audio_probe)

    p = sub.add_parser("audio-resample", help="resample audio files to mono WAV (voice prep)")
    p.add_argument("src_dir")
    p.add_argument("out_dir")
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--mono", action="store_true", default=True)
    p.set_defaults(func=cmd_audio_resample)

    p = sub.add_parser("audio-silence", help="split one audio file on silence")
    p.add_argument("audio_file")
    p.add_argument("out_dir")
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--max-sec", type=float, default=60)
    p.add_argument("--min-silence", type=float, default=0.5)
    p.add_argument("--top-db", type=float, default=40,
                   help="dB below loudest frame that counts as silence (positive)")
    p.set_defaults(func=cmd_audio_silence)

    p = sub.add_parser("audio-rvc", help="preprocess a folder into an RVC-ready dataset")
    p.add_argument("src_dir")
    p.add_argument("out_dir")
    p.add_argument("--speaker", type=int, default=0)
    p.add_argument("--max-sec", type=float, default=60)
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--top-db", type=float, default=40,
                   help="dB below loudest frame that counts as silence (positive)")
    p.set_defaults(func=cmd_audio_rvc)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
