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


def cmd_transcribe(args):
    from ai_toolset import speech

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    if args.engine == "whisper":
        result = speech.transcribe_whisper(args.audio, model=args.model,
                                           language=args.language,
                                           task=args.task, gpus=gpus)
        print(result["text"].strip())
    else:
        segments, info = speech.transcribe_faster(args.audio, model=args.model,
                                                  language=args.language,
                                                  gpus=gpus)
        print(f"[{info.language}] p={info.language_probability:.2f}")
        for line in speech.segment_lines(segments):
            print(line)
    return 0


def cmd_tts(args):
    from ai_toolset import speech

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    speech.synthesize_tts(args.text, args.out, model_name=args.model,
                          speaker_wav=args.speaker, language=args.language,
                          gpus=gpus)
    print(f"Wrote {args.out}")
    return 0


def cmd_detect(args):
    from ai_toolset import detect

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    detections, _ = detect.detect_image(args.image, weights=args.model,
                                        conf=args.conf, gpus=gpus)
    if args.json:
        print(json.dumps(detections, indent=2))
    else:
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            print(f"{det['label']:>12s}  {det['conf']:.2f}  ({x1},{y1})-({x2},{y2})")
        print(f"{len(detections)} detections")
    if args.annotate:
        out = detect.annotate(args.image, args.annotate, weights=args.model,
                              conf=args.conf, gpus=gpus)
        print(f"Annotated image: {out}")
    return 0


def cmd_detect_live(args):
    import cv2

    from ai_toolset import detect, screen

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    region = json.loads(args.region) if args.region else None
    if region is None and args.select_region:
        region = screen.select_region()
        if region is None:
            print("Cancelled.")
            return 1
    os.makedirs(args.save_dir, exist_ok=True)
    index = 0
    for frame, detections, fps in detect.detect_stream(region, weights=args.model,
                                                       conf=args.conf, gpus=gpus):
        detect.draw_detections(frame, detections)
        cv2.putText(frame, f"{fps:.1f} fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("detect-live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.imwrite(os.path.join(args.save_dir, f"{index:05d}.jpg"), frame)
            index += 1
            print(f"Saved frame {index}")
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    return 0


def cmd_train(args):
    from ai_toolset.train import best_weights, train_yolo

    gpus = args.gpus or ("0" if args.gpus is None else args.gpus)
    train_yolo(args.data, model=args.model, epochs=args.epochs,
               imgsz=args.imgsz, batch=args.batch, gpus=gpus,
               project=args.project, name=args.name, exist_ok=args.exist_ok,
               patience=args.patience)
    weights = best_weights(args.project, args.name)
    print(f"Training complete. Best weights: {weights}")
    return 0


def cmd_record_screen(args):
    import json as _json

    from ai_toolset import screen
    from ai_toolset.video import record_screen

    region = _json.loads(args.region) if args.region else None
    if region is None:
        region = screen.select_region()
        if region is None:
            print("Cancelled.")
            return 1
    frames = record_screen(region, args.out_path, duration=args.duration,
                           fps=args.fps, codec=args.codec)
    print(f"Wrote {frames} frames to {args.out_path}")
    return 0


def cmd_webcam_capture(args):
    from ai_toolset.video import webcam_capture

    recorded, snapshots = webcam_capture(args.camera, out_path=args.out_path,
                                         duration=args.duration, fps=args.fps,
                                         codec=args.codec, save_dir=args.save_dir)
    print(f"Recorded {recorded} frames, saved {snapshots} snapshots")
    return 0


def cmd_augment(args):
    from ai_toolset.images import augment_dir

    count = augment_dir(args.in_dir, args.out_dir, ext=args.ext, ops=args.ops)
    print(f"Wrote {count} augmented images")
    return 0


def cmd_record_audio(args):
    from ai_toolset.audio import record_mic

    path, seconds = record_mic(args.out, args.duration, sr=args.sr,
                               device=args.device)
    print(f"Recorded {seconds:.1f}s to {path}")
    return 0


def cmd_live_transcribe(args):
    from ai_toolset.speech import transcribe_live

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    results = transcribe_live(duration=args.duration, chunk=args.chunk,
                              model=args.model, language=args.language,
                              gpus=gpus, out_dir=args.out_dir)
    print(f"Transcribed {len(results)} segments to {args.out_dir}")
    return 0


def cmd_tts_batch(args):
    from ai_toolset.speech import synthesize_lines

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    with open(args.text_file, encoding="utf-8") as f:
        lines = f.readlines()
    written = synthesize_lines(lines, out_dir=args.out_dir,
                               model_name=args.model, speaker_wav=args.speaker,
                               language=args.language, gpus=gpus,
                               prefix=args.prefix, metadata_csv=args.metadata)
    print(f"Wrote {len(written)} wav files to {args.out_dir}")
    return 0


def cmd_narrate(args):
    from ai_toolset.speech import narrate

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    with open(args.text_file, encoding="utf-8") as f:
        lines = f.readlines()
    paths = narrate(lines, model_name=args.model, speaker_wav=args.speaker,
                    language=args.language, gpus=gpus, out_dir=args.out_dir)
    print(f"Played {len(paths)} lines")
    return 0


def cmd_benchmark(args):
    from ai_toolset.benchmark import benchmark_stt, benchmark_yolo, print_table

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    if not gpus and not args.cpu:
        from ai_toolset.cuda import detect_gpus
        gpus = [g["index"] for g in detect_gpus()] or [None]
    rows = []
    if args.audio:
        for engine in args.engines:
            rows.extend(benchmark_stt(args.audio, engine=engine,
                                      model=args.model or "base",
                                      iterations=args.iterations,
                                      gpus=gpus))
    if args.image:
        rows.extend(benchmark_yolo(args.image, weights=args.model or "yolov8n.pt",
                                   iterations=args.iterations, gpus=gpus))
    print_table(rows)
    return 0


def cmd_ocr(args):
    from ai_toolset.ocr import ocr_image, ocr_screen

    if args.image:
        text, lines = ocr_image(args.image, language=args.language)
    else:
        import json as _json

        region = _json.loads(args.region) if args.region else None
        text, lines = ocr_screen(region, language=args.language)
    print(text)
    if args.lines:
        for line in lines:
            print(f"  [{line['line_index']}] {line['text']}")
    return 0


def cmd_voice_convert(args):
    from ai_toolset.voice import convert_voice

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    out = convert_voice(args.input, args.output, args.model,
                        index_path=args.index, device=args.device,
                        f0_method=args.f0, index_rate=args.index_rate,
                        protect=args.protect, pitch=args.pitch, gpus=gpus)
    print(f"Wrote {out}")
    return 0


def cmd_diarize(args):
    from ai_toolset.diarize import diarize

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    segments = diarize(args.audio, out_path=args.out, token=args.token,
                       min_speakers=args.min_speakers,
                       max_speakers=args.max_speakers, gpus=gpus)
    for seg in segments:
        print(f"{seg['start']:7.2f}s -> {seg['end']:7.2f}s  {seg['speaker']}")
    print(f"{len(segments)} turns")
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

    p = sub.add_parser("transcribe", help="speech-to-text with whisper or faster-whisper")
    p.add_argument("audio")
    p.add_argument("--engine", choices=["whisper", "faster"], default="faster",
                   help="backend engine (default faster)")
    p.add_argument("--model", default="base",
                   help="whisper model size or HF name, e.g. tiny/base/small/medium/large-v3")
    p.add_argument("--language", help="audio language code (default: auto-detect)")
    p.add_argument("--task", choices=["transcribe", "translate"], default="transcribe",
                   help="whisper-only: translate=English output")
    p.add_argument("--gpus", help="comma-separated physical GPU indices, e.g. '0' or '0,1'")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("tts", help="text-to-speech with Coqui XTTS v2 voice cloning")
    p.add_argument("text")
    p.add_argument("out")
    p.add_argument("--model", default="tts_models/multilingual/multi-dataset/xtts_v2",
                   help="coqui model name (default XTTS v2; requires --speaker)")
    p.add_argument("--speaker",
                   help="reference wav for voice cloning (>= 6 s; XTTS requires it)")
    p.add_argument("--language", default="en", help="language for multilingual models")
    p.add_argument("--gpus", help="comma-separated physical GPU indices, e.g. '0' or '0,1'")
    p.set_defaults(func=cmd_tts)

    p = sub.add_parser("detect", help="run YOLO detection on an image")
    p.add_argument("image")
    p.add_argument("--model", default="yolov8n.pt", help="weights file or YOLO name")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--annotate", help="also write an annotated copy to this path")
    p.add_argument("--json", action="store_true", help="print detections as JSON")
    p.add_argument("--gpus", help="comma-separated physical GPU indices, e.g. '0' or '0,1'")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("detect-live", help="live YOLO detection on a screen region; s=save, q=quit")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--region", help="JSON region dict {top,left,width,height}")
    p.add_argument("--select-region", action="store_true",
                   help="interactively select the region first")
    p.add_argument("--save-dir", default="detected_frames")
    p.add_argument("--gpus", help="comma-separated physical GPU indices, e.g. '0' or '0,1'")
    p.set_defaults(func=cmd_detect_live)

    p = sub.add_parser("train", help="train a YOLO model (ultralytics)")
    p.add_argument("data", help="path to dataset data.yaml")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--gpus", default="0", help="GPU index, comma list, or '0,1'")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="detect")
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--patience", type=int, default=50)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("record-screen", help="record a screen region to a video (ESC stops)")
    p.add_argument("out_path", help="output video file, e.g. screen.mp4")
    p.add_argument("--region", help="JSON region dict {top,left,width,height}")
    p.add_argument("--duration", type=float, default=0,
                   help="max seconds to record (0 = until ESC)")
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--codec", default="mp4v")
    p.set_defaults(func=cmd_record_screen)

    p = sub.add_parser("webcam-capture",
                       help="webcam preview; r=record toggle, s=snapshot, q=quit")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out-path", help="recording destination (default webcam_<ts>.mp4)")
    p.add_argument("--duration", type=float, default=0)
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--codec", default="mp4v")
    p.add_argument("--save-dir", default="snapshots")
    p.set_defaults(func=cmd_webcam_capture)

    p = sub.add_parser("augment", help="augment images in a directory (hflip/vflip/rot/bright/etc.)")
    p.add_argument("in_dir")
    p.add_argument("out_dir", nargs="?")
    p.add_argument("--ext", default="jpg")
    p.add_argument("--ops", nargs="+",
                   help="ops to apply (default all): hflip vflip rot90 rot180 "
                        "rot270 blur bright hue")
    p.set_defaults(func=cmd_augment)

    p = sub.add_parser("record-audio",
                       help="record the mic to a mono WAV (voice cloning prep)")
    p.add_argument("out", help="output wav path")
    p.add_argument("--duration", type=float, default=0,
                   help="seconds to record (0 = until Enter)")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--device", type=int, help="sounddevice input index")
    p.set_defaults(func=cmd_record_audio)

    p = sub.add_parser("live-transcribe",
                       help="stream-transcribe the mic with faster-whisper")
    p.add_argument("--duration", type=float, default=60,
                   help="total seconds (0 = until Ctrl+C)")
    p.add_argument("--chunk", type=float, default=5, help="seconds per window")
    p.add_argument("--model", default="base")
    p.add_argument("--language", help="audio language code (default auto)")
    p.add_argument("--out-dir", default="live_segments")
    p.add_argument("--gpus", help="comma-separated physical GPU indices")
    p.set_defaults(func=cmd_live_transcribe)

    p = sub.add_parser("tts-batch",
                       help="synthesize one wav per line of a text file")
    p.add_argument("text_file")
    p.add_argument("--out-dir", default="tts_output")
    p.add_argument("--model", default="tts_models/multilingual/multi-dataset/xtts_v2")
    p.add_argument("--speaker", help="reference wav for voice cloning (XTTS)")
    p.add_argument("--language", default="en")
    p.add_argument("--prefix", default="line")
    p.add_argument("--metadata", help="also write a Coqui metadata.csv path "
                                      "(e.g. metadata.csv)")
    p.add_argument("--gpus", help="comma-separated physical GPU indices")
    p.set_defaults(func=cmd_tts_batch)

    p = sub.add_parser("narrate",
                       help="synthesize each line of a text file and play it")
    p.add_argument("text_file")
    p.add_argument("--out-dir", help="keep generated wavs in this directory")
    p.add_argument("--model", default="tts_models/multilingual/multi-dataset/xtts_v2")
    p.add_argument("--speaker", help="reference wav for voice cloning (XTTS)")
    p.add_argument("--language", default="en")
    p.add_argument("--gpus", help="comma-separated physical GPU indices")
    p.set_defaults(func=cmd_narrate)

    p = sub.add_parser("benchmark",
                       help="measure STT/YOLO latency per GPU")
    p.add_argument("--audio", help="audio file to transcribe")
    p.add_argument("--image", help="image file to run YOLO on")
    p.add_argument("--engines", nargs="+", choices=["whisper", "faster"],
                   default=["faster"])
    p.add_argument("--model", default=None,
                   help="whisper size (default: base) or YOLO weights "
                        "(default: yolov8n.pt)")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--gpus", help="comma-separated physical GPU indices "
                                  "(default: all detected)")
    p.add_argument("--cpu", action="store_true", help="also benchmark CPU")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("ocr", help="OCR an image or screen region "
                                   "(Windows.Media.Ocr)")
    p.add_argument("image", nargs="?", help="image file (omit for screen)")
    p.add_argument("--region", help="JSON region dict {top,left,width,height}")
    p.add_argument("--language", default="en")
    p.add_argument("--lines", action="store_true", help="print per-line detail")
    p.set_defaults(func=cmd_ocr)

    p = sub.add_parser("voice-convert", help="RVC voice conversion (rvc-python)")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("model", help="path to an RVC v2 .pth model")
    p.add_argument("--index", help="optional feature index .index file")
    p.add_argument("--device", help="e.g. 'cuda:0' or 'cpu' (default auto)")
    p.add_argument("--f0", default="rmvpe", choices=["rmvpe", "crepe", "pm"],
                   help="pitch extraction method")
    p.add_argument("--index-rate", type=float, default=0.75)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--pitch", type=int, default=0,
                   help="semitone shift, e.g. 12 = one octave up")
    p.add_argument("--gpus", help="comma-separated physical GPU indices")
    p.set_defaults(func=cmd_voice_convert)

    p = sub.add_parser("diarize", help="speaker diarization (pyannote, HF token)")
    p.add_argument("audio")
    p.add_argument("--out", help="write an RTTM file here")
    p.add_argument("--token", help="HF token (or HF_TOKEN env var)")
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    p.add_argument("--gpus", help="comma-separated physical GPU indices")
    p.set_defaults(func=cmd_diarize)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
