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
- Canonical sync for everything else: `uv sync --all-extras` (syncs only the
  extras you pass — plain `uv sync` prunes extra-only packages).
