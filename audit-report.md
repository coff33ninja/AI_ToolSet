# Audit Report: ai-toolset

Audit date: 2026-08-16. Focus: full project + gaps ("missed stuff") in the Streamlit UI.

## Summary

**2 high, 6 medium, 5 low findings.**

The core toolkit is healthy: pinned resolutions, no `.env` drift, no git
corruption, no stashes. The main problems are in `ai_toolset/streamlit_app.py`
— it lags the CLI/webapp feature set, has one dead cache helper, and one real
GPU bug in `transcribe_live` (`speech.py`).

---

## 1. Dependencies

| Severity | Issue | Path | Fix |
|----------|-------|------|-----|
| info | Streamlit pinned at 1.22.0 (ancient, 2022) to satisfy protobuf 3.20.3 | `pyproject.toml:120-122` | Can't bump while mediapipe 0.10.9 + TF 2.10 live in one env. Documented tradeoff. |
| info | `uv.lock` present; no `pip-audit` run | — | Run `uv run pip-audit` (not installed) |

No dependency vulnerabilities were flagged by inspection; the env resolves
cleanly (verified imports: streamlit, faster-whisper, whisper, ultralytics,
winocr, mediapipe, sounddevice, fastapi, uvicorn).

## 2. Configuration

| Severity | Issue | Path | Fix |
|----------|-------|------|-----|
| ok | `.env` / `.env.example` key sets match (HF_TOKEN, AI_TOOLSET_CUDA_RUNTIME, AI_TOOLSET_CACHE) | `.env`, `.env.example` | none |
| ok | `.gitignore` covers `.venv/`, `.uv-cache/`, `.ruff_cache/`, `.env`, `cuda_runtime/`, `yolov8n.pt` | `.gitignore` | none |
| low | Committed test artifacts pollute the repo root: `ocr_test.png`, `test_tts.wav`, `aug_out/` (6 images) | git index | Move under `smoke_data/`, add to `.gitignore`, `git rm --cached` |

## 3. Disk Usage

| Severity | Issue | Path | Size |
|----------|-------|------|------|
| medium | `.uv-cache` (project-local uv cache) holds 13.1 GB | `.uv-cache/` | 13120 MB |
| low | `.venv` is 9.7 GB (torch cu118 + TF 2.10 + mediapipe wheels) | `.venv/` | 9726 MB |

Both are on the workspace drive (E:), git-ignored, by design. `.uv-cache` can
be pruned with `uv cache clean` (it re-downloads on demand).

## 4. Environment Portability

| Severity | Issue | Path | Fix |
|----------|-------|------|-----|
| ok | Everything is project-local: venv, uv cache, vendored fairseq, `cuda_runtime/` | — | none |
| ok | `load_env()` wired into CLI, web, and ui entry points | `__main__.py:798-800`, `streamlit_app.py:14-16` | none |
| low | `__main__.py` imports `cv2` at module top | `__main__.py:8` | Lazy-import to make `--help` cheap |

## 5. Git Health

| Severity | Issue | Path | Fix |
|----------|-------|------|-----|
| medium | 2 uncommitted feature changes: `speech.py` (+49), `streamlit_app.py` (+39) — `ensure_tts_model`/`_tts_cache_dir` and streamlit diarize-guard + keyed uploaders | working tree | commit or stash |
| low | Stale remote branches: `origin/dependabot/uv/uv-90f8f7485e`, `origin/imgbot` | git | delete if merged/abandoned |
| ok | `git fsck --no-dangling` clean, no stashes | — | none |

---

## 6. Streamlit UI — missed / broken (the focus)

### High

- **Dead cache helper, model reloaded on every click.** `load_whisper()`
  (`streamlit_app.py:47-51`) is never called. The STT tab calls
  `transcribe_faster(...)` directly (`streamlit_app.py:76-78`), which builds a
  fresh `WhisperModel` per transcription. And `load_whisper` itself hardcodes
  `device="auto"` — it would neither honor the sidebar GPU picker nor run the
  Pascal-aware `compute_type` auto-detection that `transcribe_faster` does.
  Fix: call `load_whisper` (route through `transcribe_faster`, pass `gpus`).

- **`transcribe_live` silently ignores GPU selection.** `speech.py:251` hardcodes
  `device="cpu", compute_type="int8"`. The CLI `live-transcribe --gpus` is
  accepted but dropped; faster-whisper's CUDA is never used. Fix: auto-detect
  like `transcribe_faster` (respect `gpus` + `ctranslate2` compute types).

### Medium (CLI features absent from the UI)

| Missed in Streamlit | CLI equivalent | Why it matters |
|---------------------|----------------|----------------|
| Whisper `large-v3`, language, engine (faster/openai), translate task | `transcribe` | UI caps at `medium`, no `--language`, no `--task` |
| TTS language selector | `tts --language` | XTTS v2 is multilingual; UI hardcodes `en` |
| TTS batch / narration | `tts-batch`, `narrate` (`synthesize_lines`) | No multi-line → multi-wav flow in the UI |
| Video detection / live detection / webcam | `detect-live`, `detect_stream` | UI is image-only |
| OCR language + screen OCR | `ocr --language`, `ocr_screen` | UI hardcodes `en`, no screen capture |
| MediaPipe tab | `mediapipe` (`webcam_stream`) | Whole webcam-landmark feature absent |
| Benchmark tab | `benchmark` (`benchmark_stt`, `benchmark_yolo`) | FastAPI dashboard has it; Streamlit doesn't |
| Mic recording / live transcription | `record-audio`, `live-transcribe` | Voice-cloning prep absent |

### Low

- **Model-downloads tab is hardcoded and incomplete.** Fetches only
  `yolov8n.pt`, whisper `base`, tacotron2-DDC. Misses: whisper `large-v3`
  (and other sizes), the other 3 TTS models offered on the TTS tab,
  RVC validation (`models.ensure_rvc`), and a status table (`models.summarize`).
- **Temp-file leak on the TTS failure path.** `streamlit_app.py:99-113` uses
  `delete=False` for the output + speaker wavs and only `os.remove` after a
  successful `st.audio`. An exception in `synthesize_tts` leaks both files.
  Fix: wrap in try/finally.
- **README overstates parity.** `README.md:246` claims the Streamlit app is
  "the same feature set in a lighter wrapper" — the FastAPI dashboard has
  benchmark + webcam/mediapipe tabs that Streamlit lacks.

---

## 7. Follow-up — fixes applied (2026-08-16)

Resolved after this report:

- `speech.py`: added `load_faster_model()` (shared GPU + Pascal-aware
  compute-type detection); `transcribe_faster` and `transcribe_live` now use
  it, so `live-transcribe --gpus` actually uses the GPU (was hardcoded
  CPU/int8). `transcribe_live` + `benchmark` benefit too.
- `streamlit_app.py`: STT tab now uses the cached loader (no reload per
  click), adds `large-v3`, language, engine, and translate task. TTS tab adds
  a language selector + batch synthesize (`synthesize_lines`) with try/finally
  temp cleanup. OCR tab adds language + screen OCR. Detection tab adds video
  sampling + annotated-image download. New tabs: Benchmark (STT+YOLO),
  MediaPipe (camera_input overlay), Audio capture (mic record + live
  transcription).
- `README.md`: corrected the Streamlit parity wording.
- Repo hygiene: `ocr_test.png`, `test_tts.wav`, `aug_out/` untracked
  (`git rm --cached`) and added to `.gitignore`; `uv cache prune` freed
  ~1 GB from `.uv-cache`.
- Not done: remote branches `origin/dependabot/uv/uv-90f8f7485e` and
  `origin/imgbot` are unmerged — left untouched pending review.
