"""FastAPI dashboard for the ai-toolset: OCR, detection, STT, TTS, benchmark,
and a live MediaPipe webcam stream.

Run with:  uv run python -m ai_toolset web [--host 127.0.0.1 --port 8000]

Heavy backends (YOLO, whisper, coqui TTS, mediapipe) are imported lazily and
cached, so the dashboard works even when only part of the toolkit's extras
are installed - the health endpoint reports exactly what is available.
"""

import argparse
import base64
import io
import os
import sys
import tempfile

from ai_toolset.cuda import detect_gpus

STATIC_DIR = os.path.join(os.path.dirname(__file__), "web")
DEFAULT_TTS = "tts_models/en/ljspeech/tacotron2-DDC"
DEFAULT_WEIGHTS = "yolov8n.pt"
DEFAULT_WHISPER = "base"

_APP = None


def get_app(gpus=None):
    """Create (once) and return the FastAPI app."""
    global _APP
    if _APP is None:
        _APP = create_app(gpus=gpus)
    return _APP


def _extras_status():
    probes = {
        "voice": "torch",
        "vision": "ultralytics",
        "stt": "faster_whisper",
        "tts": "TTS",
        "ocr": "winocr",
        "mediapipe": "mediapipe",
        "rvc": "rvc_python",
        "diarize": "pyannote.audio",
        "web": "fastapi",
        "ui": "streamlit",
    }
    # coqui TTS needs the numpy<2 compat shim before it can import.
    from ai_toolset.speech import _coqui_numpy_compat

    _coqui_numpy_compat()
    return {name: _importable(mod) for name, mod in probes.items()}


def _importable(module):
    import importlib

    try:
        importlib.import_module(module)
        return True
    except Exception:  # noqa: BLE001 - any import failure means "not usable"
        return False


def _gpus():
    return detect_gpus()


def _yolo_model(weights):
    from ai_toolset.detect import _load

    return _load(weights)


def _whisper_model(model):
    from faster_whisper import WhisperModel

    return WhisperModel(model, device="auto")


def _tts_model(model_name):
    from ai_toolset.speech import _coqui_numpy_compat

    _coqui_numpy_compat()
    from TTS.api import TTS

    return TTS(model_name, gpu=False)


def _save_upload(upload, suffix):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(upload.file.read())
    return path


def create_app(gpus=None):
    from fastapi import FastAPI, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="AI ToolSet", version="0.1.0")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def _gpus_from(sel):
        if sel is None:
            return gpus
        return [int(x) for x in sel.split(",") if x.strip()]

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/api/health")
    def health():
        return {
            "gpus": _gpus(),
            "extras": _extras_status(),
            "webcam": _camera_open(0),
            "tts_default": DEFAULT_TTS,
            "whisper_default": DEFAULT_WHISPER,
            "weights_default": DEFAULT_WEIGHTS,
        }

    @app.get("/api/cameras")
    def cameras():
        found = []
        for i in range(4):
            if _camera_open(i):
                found.append(i)
        return {"cameras": found}

    @app.post("/api/ocr")
    def ocr(image: UploadFile | None = None, screen: bool = Form(False),
            language: str = Form("en")):
        from ai_toolset.ocr import ocr_image, ocr_screen

        try:
            if screen:
                text, lines = ocr_screen(language=language)
            elif image is None:
                raise HTTPException(400, "send an image file or screen=true")
            else:
                path = _save_upload(image, os.path.splitext(image.filename or "x")[1])
                try:
                    text, lines = ocr_image(path, language=language)
                finally:
                    os.remove(path)
            return {"text": text, "lines": lines}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    @app.post("/api/detect")
    def detect(image: UploadFile, weights: str = Form(DEFAULT_WEIGHTS),
               conf: float = Form(0.25), gpus_sel: str | None = Form(None)):
        from ai_toolset.detect import detect_frame, draw_detections

        path = _save_upload(image, os.path.splitext(image.filename or "x")[1])
        try:
            import cv2

            frame = cv2.imread(path)
            if frame is None:
                raise HTTPException(400, f"could not decode image {image.filename}")
            model = _yolo_model(weights)
            detections = detect_frame(frame, weights=weights, conf=conf,
                                      model=model, gpus=_gpus_from(gpus_sel))
            draw_detections(frame, detections)
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                raise HTTPException(500, "could not encode annotated image")
            return {"detections": detections,
                    "annotated_b64": base64.b64encode(buf.tobytes()).decode()}
        finally:
            os.remove(path)

    @app.post("/api/transcribe")
    def transcribe(audio: UploadFile, model: str = Form(DEFAULT_WHISPER),
                   language: str | None = Form(None),
                   gpus_sel: str | None = Form(None)):
        from ai_toolset.speech import transcribe_faster

        path = _save_upload(audio, ".wav")
        try:
            segments, info = transcribe_faster(
                path, model=model, language=language, gpus=_gpus_from(gpus_sel))
            return {
                "text": "\n".join(s.text.strip() for s in segments),
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": [{"start": s.start, "end": s.end, "text": s.text.strip()}
                             for s in segments],
            }
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc
        finally:
            os.remove(path)

    @app.post("/api/tts")
    def tts(text: str = Form(...), model: str = Form(DEFAULT_TTS),
            language: str = Form("en"), speaker: UploadFile | None = None,
            gpus_sel: str | None = Form(None)):
        from ai_toolset.speech import synthesize_tts

        speaker_path = None
        if speaker is not None:
            speaker_path = _save_upload(speaker, ".wav")
        try:
            fd, out = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            synthesize_tts(text, out, model_name=model, language=language,
                           speaker_wav=speaker_path, gpus=_gpus_from(gpus_sel))
            with open(out, "rb") as f:
                data = f.read()
            os.remove(out)
            return StreamingResponse(io.BytesIO(data),
                                     media_type="audio/wav")
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc
        finally:
            if speaker_path and os.path.exists(speaker_path):
                os.remove(speaker_path)

    @app.post("/api/benchmark")
    def benchmark(image: UploadFile | None = None,
                  gpus_sel: str | None = Form(None),
                  iterations: int = Form(3)):
        from ai_toolset.benchmark import benchmark_stt, benchmark_yolo

        sel = _gpus_from(gpus_sel) or [None]
        rows = []
        if image is not None:
            path = _save_upload(image, os.path.splitext(image.filename or "x")[1])
            try:
                rows.extend(benchmark_yolo(path, weights=DEFAULT_WEIGHTS,
                                           iterations=iterations, gpus=sel))
            finally:
                os.remove(path)
        else:
            rows.extend(benchmark_stt(None, engine="faster",
                                      model=DEFAULT_WHISPER,
                                      iterations=iterations, gpus=sel))
        return {"rows": rows}

    @app.get("/api/webcam.mjpg")
    def webcam(camera: int = 0, solution: str = "pose", mirror: int = 1,
               quality: int = 80):
        if solution not in ("none", "pose", "hands", "face", "holistic", "selfie"):
            raise HTTPException(400, f"bad solution '{solution}'")
        import cv2

        from ai_toolset.mp import available as mp_available

        if solution != "none" and not mp_available():
            raise HTTPException(400, "mediapipe extra not installed")
        from ai_toolset.mp import overlay as mp_overlay

        def stream():
            cap = cv2.VideoCapture(camera)
            if not cap.isOpened():
                raise HTTPException(500, f"could not open camera {camera}")
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if solution != "none":
                        frame = mp_overlay(frame, solution=solution,
                                           mirror=bool(mirror))
                    elif mirror:
                        frame = cv2.flip(frame, 1)
                    ok, buf = cv2.imencode(".jpg", frame,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                    if not ok:
                        continue
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buf.tobytes() + b"\r\n")
            finally:
                cap.release()

        return StreamingResponse(
            stream(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app


def _camera_open(index):
    import cv2

    try:
        cap = cv2.VideoCapture(index)
        ok = cap.isOpened()
        cap.release()
        return ok
    except (cv2.error, OSError):
        return False


def main(argv=None):
    import webbrowser

    import uvicorn

    from ai_toolset.env import load_env

    load_env()

    parser = argparse.ArgumentParser(
        prog="python -m ai_toolset web",
        description="Serve the AI ToolSet web dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="open the browser")
    parser.add_argument("--gpus", help="comma-separated GPU indices used by default")
    args = parser.parse_args(argv)

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    app = create_app(gpus=gpus)
    if args.open:
        webbrowser.open(f"http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
