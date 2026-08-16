# AI ToolSet TODO

Tracker for the ai-toolset project. `[x]` = implemented and verified in the
repo; `[ ]` = planned or not yet verified. Update this file as work lands.

## Completed

- [x] STT: openai-whisper + faster-whisper GPU transcription (`transcribe`)
- [x] TTS: Coqui XTTS v2 voice cloning / single-speaker synthesis (`tts`)
- [x] faster-whisper compute-type auto-detection (Pascal has no fp16)
- [x] Project-local uv cache on the workspace drive (`.uv-cache`, link-mode copy)

## Incomplete

### Vision
- [ ] YOLO training (ultralytics) — `train` command
- [ ] YOLO detection — `detect` command (image / video / region)
- [ ] Live screen detection — `detect-live` (YOLO on the screen stream, overlay + FPS)
- [ ] Screen recording — `record-screen` (mp4 of a screen region)
- [ ] Webcam capture — `webcam-capture` (record frames / video from camera)
- [ ] Image augmentation — `augment` (flip / rotate / color jitter etc.)

### Speech
- [ ] RVC voice conversion — `voice-convert` (rvc-python)
- [ ] Audio capture — `record-audio` (mic → wav for XTTS/RVC prep)
- [ ] Live transcription — `live-transcribe` (mic stream → faster-whisper)
- [ ] Diarization — `diarize` (pyannote, needs HF token + gated model)
- [ ] TTS batch / narration — `tts-batch` and `narrate` (read a text file aloud)

### OCR
- [ ] OCR — `ocr` (Windows.Media.Ocr via winocr, no model downloads)

### Utilities
- [ ] Benchmark — `benchmark` (per-engine/per-GPU latency)
- [ ] TTS fine-tune dataset builder — `tts-dataset` (text lines → wavs + metadata.csv)

## Notes
- New deps are project-local extras (uv), never system installs.
- Risk: rvc-python / pyannote are heavy graphs (fairseq, hydra, speechbrain);
  keep them in separate extras so a failure can't break the base lock.
