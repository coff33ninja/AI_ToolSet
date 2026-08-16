"""End-to-end smoke test for the ai-toolset.

Replays the CLI battery verified during development: per-extra imports, GPU
detection, YOLO detect + annotate + augment + benchmark, OCR, MediaPipe
overlay, batch TTS, and the diarize token error path. Prints PASS/FAIL per
step and exits 1 if anything fails.

Run:  uv run python examples/smoke_test.py
"""

import json
import os
import sys
import tempfile
import wave

import cv2
import numpy as np


def step(name):
    def deco(fn):
        steps.append((name, fn))
        return fn

    return deco


steps = []
results = []


def run_all():
    for name, fn in steps:
        try:
            fn()
            results.append((name, "PASS", ""))
        except Exception as exc:  # noqa: BLE001 - test harness
            results.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))

    width = max(len(n) for n, _, _ in results)
    for name, status, err in results:
        print(f"[{status:4}] {name:<{width}}  {err}")
    ok = all(s == "PASS" for _, s, _ in results)
    print(f"\n{sum(1 for _, s, _ in results if s == 'PASS')}/{len(results)} passed")
    return 0 if ok else 1


@step("imports: base (numpy, cv2, tensorflow, keras)")
def _imports_base():
    import cv2  # noqa: F401
    import keras  # noqa: F401
    import numpy  # noqa: F401
    import tensorflow  # noqa: F401


@step("imports: ocr (winocr)")
def _imports_ocr():
    import winocr  # noqa: F401


@step("imports: mediapipe")
def _imports_mp():
    import mediapipe  # noqa: F401


@step("imports: web (fastapi, uvicorn)")
def _imports_web():
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401


@step("imports: vision (ultralytics)")
def _imports_vision():
    import ultralytics  # noqa: F401


@step("imports: stt (faster-whisper)")
def _imports_stt():
    import faster_whisper  # noqa: F401


@step("faster-whisper load_faster_model (device/compute-type auto)")
def _load_faster():
    from ai_toolset.cuda import detect_gpus
    from ai_toolset.speech import load_faster_model

    gpus = [g["index"] for g in detect_gpus()]
    model = load_faster_model("tiny", gpus=gpus or None)
    device = model.model.device
    assert device == ("cuda" if gpus else "cpu"), (
        f"expected {'cuda' if gpus else 'cpu'}, got {device}"
    )
    print(f"        {device}/{model.model.compute_type}")


@step("faster-whisper transcribe_faster on synthetic tone")
def _transcribe_faster():
    from ai_toolset.speech import transcribe_faster

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tone.wav")
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5))
        data = (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((data * 32767).astype("<i2").tobytes())
        segments, info = transcribe_faster(path, model="tiny", gpus=None)
        assert info.language, "no language detected"
        print(f"        {info.language} {len(segments)} segment(s)")


@step("imports: tts (coqui TTS)")
def _imports_tts():
    from ai_toolset.speech import _coqui_numpy_compat

    _coqui_numpy_compat()
    from TTS.api import TTS  # noqa: F401


@step("imports: rvc (rvc-python, fairseq)")
def _imports_rvc():
    import rvc_python  # noqa: F401


@step("gpu detection via nvidia-smi")
def _gpus():
    from ai_toolset.cuda import detect_gpus

    gpus = detect_gpus()
    assert gpus, "no NVIDIA GPU detected"
    print(f"        {[(g['index'], g['name']) for g in gpus]}")


@step("yolo detect + annotate on synthetic image")
def _yolo():
    from ai_toolset import detect

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.jpg")
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        cv2.rectangle(img, (60, 60), (200, 200), (0, 0, 255), -1)
        cv2.imwrite(path, img)
        detections, _ = detect.detect_image(path)
        out = detect.annotate(path, os.path.join(tmp, "out.jpg"))
        assert os.path.isfile(out)
        print(f"        {len(detections)} detections -> {os.path.basename(out)}")


@step("augment synthetic image")
def _augment():
    from ai_toolset.images import augment_dir

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        dst = os.path.join(tmp, "dst")
        os.makedirs(src)
        cv2.imwrite(os.path.join(src, "a.jpg"), np.zeros((64, 64, 3), dtype=np.uint8))
        count = augment_dir(src, dst, ops=["hflip", "vflip", "rot90", "bright"])
        assert count > 0


@step("yolo benchmark (1 iteration)")
def _benchmark():
    from ai_toolset.benchmark import benchmark_yolo
    from ai_toolset.cuda import detect_gpus

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.jpg")
        cv2.imwrite(path, np.zeros((320, 320, 3), dtype=np.uint8))
        gpus = [g["index"] for g in detect_gpus()] or [None]
        rows = benchmark_yolo(path, iterations=1, gpus=gpus)
        assert rows
        print(f"        {rows[0]['mean_ms']:.1f} ms mean (gpu {rows[0]['gpus']})")


@step("ocr on synthetic text image")
def _ocr():
    from ai_toolset.ocr import ocr_image

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "text.png")
        img = np.zeros((160, 640, 3), dtype=np.uint8)
        cv2.putText(
            img,
            "Hello AI Toolset 2026",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
        )
        cv2.imwrite(path, img)
        text, _ = ocr_image(path)
        assert "Hello" in text or "AI Toolset" in text, f"got: {text!r}"
        print(f"        read {text!r}")


@step("mediapipe pose overlay on synthetic frame")
def _mp():
    from ai_toolset.mp import overlay

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = overlay(frame, solution="pose", static=True)
    assert out.shape == frame.shape


@step("webapp: create_app + /api/health over live uvicorn")
def _webapp():
    import socket
    import threading
    import time
    import urllib.request

    import uvicorn

    from ai_toolset.webapp import create_app

    app = create_app()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}/api/health"
        last = None
        for _ in range(50):
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    payload = json.load(r)
                break
            except Exception as exc:  # noqa: BLE001 - server not up yet
                last = exc
                time.sleep(0.2)
        else:
            raise AssertionError(f"server did not answer: {last}")
        assert "gpus" in payload and "extras" in payload
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@step("tts-batch with cached tacotron2-DDC")
def _tts_batch():
    from ai_toolset.speech import synthesize_lines

    with tempfile.TemporaryDirectory() as tmp:
        txt = os.path.join(tmp, "lines.txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write("First line of the fine tune set.\n")
        with open(txt, encoding="utf-8") as f:
            lines = f.read().splitlines()
        written = synthesize_lines(
            lines,
            out_dir=os.path.join(tmp, "out"),
            model_name="tts_models/en/ljspeech/tacotron2-DDC",
        )
        assert written and os.path.isfile(written[0])
        print(f"        {len(written)} wav written")


@step("diarize without HF token raises friendly error")
def _diarize_error():
    from ai_toolset.diarize import diarize

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "x.wav")
        with open(wav, "wb") as f:
            f.write(b"RIFF")
        try:
            diarize(wav)
        except RuntimeError as exc:
            assert "Hugging Face token" in str(exc)
            return
        raise AssertionError("expected RuntimeError")


def main():
    sys.exit(run_all())


if __name__ == "__main__":
    main()
