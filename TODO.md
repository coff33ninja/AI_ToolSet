# AI ToolSet TODO

Tracker for the ai-toolset project. `[x]` = implemented and verified in the
repo; `[ ]` = planned or not yet verified. Update this file as work lands.

## Completed

- [x] STT: openai-whisper + faster-whisper GPU transcription (`transcribe`)
- [x] TTS: Coqui XTTS v2 voice cloning / single-speaker synthesis (`tts`)
- [x] faster-whisper compute-type auto-detection (Pascal has no fp16)
- [x] Project-local uv cache on the workspace drive (`.uv-cache`, link-mode copy)
- [x] YOLO training (ultralytics) — `train` command
- [x] YOLO detection — `detect` command (image / video / region)
- [x] Live screen detection — `detect-live` (YOLO on the screen stream, overlay + FPS)
- [x] Screen recording — `record-screen` (mp4 of a screen region)
- [x] Webcam capture — `webcam-capture` (record frames / video from camera)
- [x] Image augmentation — `augment` (flip / rotate / color jitter etc.)
- [x] Audio capture — `record-audio` (mic → wav for XTTS/RVC prep)
- [x] Live transcription — `live-transcribe` (mic stream → faster-whisper)
- [x] TTS batch / narration — `tts-batch` and `narrate` (read a text file aloud)
- [x] TTS fine-tune dataset builder — folded into `tts-batch --metadata metadata.csv`
- [x] OCR — `ocr` (Windows.Media.Ocr via winocr, no model downloads)
- [x] Benchmark — `benchmark` (per-engine/per-GPU latency)
- [x] MediaPipe — `mediapipe` live landmark/selfie overlay CLI + `mp` module
      (pose/hands/face/holistic/selfie; verified end-to-end in smoke test)
- [x] Web dashboard — `web` FastAPI dashboard + `webapp` module + `ai_toolset/web/`
      frontend (status/OCR/detect/transcribe/TTS/benchmark/webcam + GPU select)
- [x] Streamlit quick-UI — `ui` command + `streamlit_app.py` (STT/TTS/Detect/OCR/
      model downloads tabs)
- [x] Model downloader — `get-models` command + `models` module (yolo/whisper/tts/
      diarize/rvc status + download)
- [x] Smoke test — `examples/smoke_test.py` (17-step battery, 17/17 passing)

## Incomplete

### Speech
- [ ] RVC voice conversion — `voice-convert` implemented; needs a user `.pth`
      model + index to verify end-to-end (import verified)
- [ ] Diarization — `diarize` implemented; needs HF token + gated model +
      gated-license acceptance to run (import + token check verified)

## Notes
- New deps are project-local extras (uv), never system installs.
- `rvc` and `diarize` are declared **conflicting** extras in `[tool.uv]`: fairseq
  (rvc) needs `hydra-core<1.1` → `omegaconf<2.1`, the pyannote stack (diarize)
  needs `omegaconf>=2.1`, so they cannot share one environment. Install one
  profile: `uv sync --extra rvc` **or** `uv sync --extra diarize`.
- Canonical sync for everything except the conflicting pair:
  `uv sync --extra rvc --extra voice --extra vision --extra audio --extra ocr
  --extra stt --extra tts --extra mediapipe --extra web --extra ui`.
- MediaPipe pins (verified): `mediapipe==0.10.9`, `protobuf==3.20.3`,
  `opencv-contrib-python==4.10.0.84` + `opencv-python==4.10.0.84` both
  overridden. 0.10.31+ wheels ship only the `tasks` API; 0.10.14+ needs
  protobuf>=4.25 which breaks TF 2.10; protobuf 3.20.3 is the only wheel that
  still ships `internal/builder.py` that mediapipe's generated pb2 imports.
