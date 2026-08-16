# AI ToolSet

Reusable toolkit for GPU-accelerated machine learning projects on Windows. It
solves the two things every local TF/PyTorch project trips over:

1. **CUDA works without guessing** - `scripts/get_cuda_runtime.ps1` figures
   out which CUDA/cuDNN runtime your TensorFlow version needs, downloads the
   exact builds into a project-local `cuda_runtime/` folder, and wires it up.
   No system-wide CUDA toolkit install, no version guessing, no broken GPU.
   The same script covers **PyTorch voice cloning** with a driver-only check
   (torch wheels bundle their own CUDA/cuDNN - nothing is installed globally).
2. **A grab-bag of reusable helpers** - screen capture, video frame
   extraction/stitching, synthetic detection datasets, image/label
   utilities, and **voice-cloning dataset prep**, all importable from one
   package.

The pattern was extracted from
[poke.AI](https://github.com/coff33ninja/poke.AI), which runs TensorFlow 2.10
with native Windows GPU support via this exact mechanism.

## Why project-local CUDA?

TensorFlow 2.10 is the last TF with native Windows GPU support. It needs the
CUDA 11.2 + cuDNN 8.1 runtime DLLs on `PATH`, but you do not need (and
should not want) a full system CUDA install just to run training. This
toolkit keeps the DLLs in `cuda_runtime/bin` and prepends that folder to
`PATH` at every interpreter start via `sitecustomize.py`.

Nothing is guessed:

- The TensorFlow -> CUDA/cuDNN mapping is a verified table (below).
- The TensorFlow -> CUDA/cuDNN mapping is a verified table (below).
- The CUDA runtime DLLs (cudart, cublas, cufft, curand, cusolver, cusparse)
  come from NVIDIA's own redistributable wheels on PyPI (`nvidia-*-cu11`),
  pinned to known-good versions.
- cuDNN always comes from the **official NVIDIA redistributable** server
  (developer.download.nvidia.com) - third-party builds of cuDNN crash
  TensorFlow 2.10 with `0xC0000409` on the first GPU op.
- Your NVIDIA driver version is checked against the CUDA minimum and you get
  a warning if it is too old.

| TensorFlow | CUDA | cuDNN | min. driver |
|------------|------|-------|-------------|
| 2.4        | 11.0 | 8.0.5.39 | 451.82 |
| 2.5 - 2.10 | 11.2 | 8.1.0.77 | 460.89 |

## Quickstart

Prerequisites: Windows 10/11, an NVIDIA GPU (9-series+), updated drivers,
[uv](https://docs.astral.sh/uv/), and Git.

```powershell
uv sync                                      # creates .venv (Python 3.10 + TF 2.10)
powershell -ExecutionPolicy Bypass -File scripts/get_cuda_runtime.ps1
powershell -ExecutionPolicy Bypass -File scripts/verify_cuda.ps1 -RunTensorFlowCheck
```

Most features are opt-in extras so the base install stays lean. To enable
everything at once (mediapipe/web/ui included; `rvc` and `diarize` conflict
and cannot coexist):

```powershell
uv sync --extra rvc --extra voice --extra vision --extra audio --extra ocr `
       --extra stt --extra tts --extra mediapipe --extra web --extra ui
```

The last command prints your GPU, driver, runtime DLL status, and whether
TensorFlow actually sees the GPU. Expected output ends with:

```
[PhysicalDevice(name="/physical_device:GPU:0", device_type="GPU")]
```

For a different TensorFlow version, pass it through and the script resolves
the matching runtime:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/get_cuda_runtime.ps1 -TensorFlowVersion 2.4
```

Or override the runtime explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/get_cuda_runtime.ps1 -CudaVersion 11.2 -CudnnVersion 8.1.0.77
```

## Voice cloning (PyTorch stack)

The same project-local philosophy covers voice cloning and TTS (RVC,
XTTS v2, so-vits-svc). Everything stays inside `.venv` - **never a global
install**.

PyTorch differs from TensorFlow on Windows: its CUDA wheels **bundle** the
CUDA + cuDNN runtime inside `torch/lib`, so there is nothing to download.
The only system requirement is a recent NVIDIA driver. Install the voice
extra, then let the script verify:

```powershell
uv sync --extra voice                                     # torch + torchaudio + librosa + soundfile into .venv
powershell -ExecutionPolicy Bypass -File scripts/get_cuda_runtime.ps1 -Framework pytorch
powershell -ExecutionPolicy Bypass -File scripts/verify_cuda.ps1 -CheckTorch
```

`get_cuda_runtime.ps1 -Framework pytorch` checks your driver against the
CUDA 11.8 minimum (452.39), downloads nothing, and reports whether the
installed torch sees the GPU. `torch`/`torchaudio` resolve from the official
`download.pytorch.org/whl/cu118` index via `[tool.uv.sources]` - the cu118
wheels keep Pascal-and-newer GPUs (GTX 10-series included) working.

### Dataset prep

RVC/so-vits-svc want a folder of clean, resampled speech segments before
training. The toolkit automates that:

```powershell
uv run python -m ai_toolset audio-probe RAW/                       # see what you have
uv run python -m ai_toolset audio-resample RAW/ CLEAN/ --sr 48000  # -> 48 kHz mono WAV
uv run python -m ai_toolset audio-silence clip.wav SEGS/           # split one file on silence
uv run python -m ai_toolset audio-rvc CLEAN/ RVC_DATA/ --speaker 0 # -> 0_000000.wav ...
```

or as one pipeline:

```powershell
uv run python examples/prepare_voice_dataset.py RAW/ RVC_DATA/ --speaker 0
```

`audio-rvc` splits each recording on silence (merging short gaps, hard-capping
segments at `--max-sec`) and writes `<speaker>_<index>.wav`, the naming the
RVC webui expects for preprocessing. Point the webui's dataset folder at the
output and its training tab at your model folder - both run locally from your
venv.

## Speech-to-text and text-to-speech

Same philosophy, two more extras. **Everything stays inside `.venv`.**

```powershell
uv sync --extra stt              # openai-whisper + faster-whisper + NVIDIA CUDA 12 runtime
uv sync --extra tts              # coqui-tts (XTTS v2, tacotron2, ...)
powershell -ExecutionPolicy Bypass -File scripts/get_cuda_runtime.ps1 -Framework fasterwhisper
```

### STT engines

- **whisper** (OpenAI, `transcribe_whisper`): runs on the cu118 torch from the
  `voice` extra. Higher quality, slower.
- **faster-whisper** (CTranslate2, `transcribe_faster`): CTranslate2 is the
  default engine - a lean transformer runtime with a VAD filter and ~4x
  faster decoding. Same model names (`tiny`, `base`, `small`, `medium`,
  `large-v3`), auto-fp16 on GPU, `int8` on CPU.

For GPU, CTranslate2's official wheels already bundle cuDNN 9
(`cudnn64_9.dll` sits inside the wheel). The only missing Windows runtime
pieces are the CUDA 12 cuBLAS + cudart DLLs, so the `stt` extra installs
**NVIDIA's own redistributable wheels** (`nvidia-cublas-cu12`,
`nvidia-cuda-runtime-cu12`) into `.venv`, and `sitecustomize.py` puts their
`site-packages/nvidia/*/lib` dirs on `PATH` at interpreter startup. No system
CUDA install, no cuDNN download, no conda. CPU works out of the box.

```powershell
uv run python -m ai_toolset transcribe audio.wav --engine faster --model base --language en --gpus 0
uv run python -m ai_toolset transcribe audio.wav --engine whisper --model base --gpus 0
uv run python examples/transcribe_audio.py audio.wav
```

### TTS (voice cloning)

`coqui-tts` runs on the same torch. XTTS v2 clones a voice from one reference
clip; smaller English models (ljspeech/tacotron2-DDC) need none:

```powershell
uv run python -m ai_toolset tts "Hello from the toolset." out.wav --speaker ref.wav
uv run python -m ai_toolset tts "Hello from the toolset." out.wav --model tts_models/en/ljspeech/tacotron2-DDC
uv run python examples/synthesize_speech.py "Hello" out.wav
```

### Batch TTS / narration / fine-tune dataset

Synthesize one wav per line of a text file, with an optional Coqui-format
`path|text` metadata file for fine-tuning:

```powershell
uv sync --extra tts
uv run python -m ai_toolset tts-batch lines.txt --out-dir wavs --metadata metadata.csv --gpus 0
uv run python -m ai_toolset narrate lines.txt --gpus 0      # synthesize + play each line
```

## Detection, OCR, and live capture

Another batch of extras, same project-local rules:

```powershell
uv sync --extra vision --extra ocr     # ultralytics + torchvision; winocr
```

- `detect` / `detect-live` - YOLO on an image/video or a live screen region
  (overlay + FPS, `s` saves the frame, `q` quits).
- `train` - ultralytics YOLO training on your dataset dir.
- `record-screen` - record a screen region to mp4 (ESC stops).
- `webcam-capture` - webcam preview; `r` toggles recording, `s` snapshots.
- `augment` - flip/rotate/color-jitter an image directory (dataset boost).
- `ocr` - Windows.Media.Ocr via `winocr` (no model downloads), works on an
  image file or a live screen region.

```powershell
uv run python -m ai_toolset detect img.png --gpus 0
uv run python -m ai_toolset detect-live --select-region --gpus 0
uv run python -m ai_toolset train data.yaml --gpus 0 --epochs 50
uv run python -m ai_toolset record-screen out.mp4 --select-region --fps 30
uv run python -m ai_toolset webcam-capture --camera 0
uv run python -m ai_toolset augment data/images out/ --per-image 5
uv run python -m ai_toolset ocr screenshot.png --language en
```

## MediaPipe landmarks

Live face/hands/pose/holistic landmarks and selfie segmentation on webcam
frames, via the mediapipe `solutions` API:

```powershell
uv sync --extra mediapipe
uv run python -m ai_toolset mediapipe --camera 0 --solution pose   # s=save, q=quit
```

`--solution` accepts `pose`, `hands`, `face`, `holistic`, or `selfie`.
MediaPipe 0.10.9 is pinned: later 0.10.x wheels ship only the `tasks` API
(whose `.task` model files are separate downloads), and 0.10.14+ requires
protobuf 4.x, which TensorFlow 2.10 cannot run with. Its `opencv-contrib`
dependency is forced to the shared 4.10.0.84 pin.

## Web dashboard and quick UI

Two frontends over the same underlying modules, both fully local.

```powershell
uv sync --extra web                # FastAPI dashboard
uv sync --extra ui                 # Streamlit quick-UI
```

```powershell
uv run python -m ai_toolset web    # serves http://127.0.0.1:8000 (opens browser)
uv run python -m ai_toolset ui     # launches Streamlit (opens browser)
```

The FastAPI dashboard (`webapp.py` + `ai_toolset/web/`) has tabs for status,
OCR, YOLO detection, transcription, TTS, latency benchmarks, and a webcam
stream with a mediapipe overlay dropdown. GPU multi-select drives every
model call. The Streamlit app (`streamlit_app.py`) covers the same core
features in a lighter wrapper — STT/TTS/detection/OCR plus latency
benchmarks, a MediaPipe webcam overlay, mic recording / live transcription,
batch TTS, video detection, and a model-downloads tab.

### Model downloads

`models.py` centralizes the downloads that are not automatic, and reports
status for the ones that are:

```powershell
uv run python -m ai_toolset get-models              # status table
uv run python -m ai_toolset get-models --yolo       # ultralytics weights (auto at first use)
uv run python -m ai_toolset get-models --whisper base
uv run python -m ai_toolset get-models --tts
uv run python -m ai_toolset get-models --diarize    # needs a gated HF token
uv run python -m ai_toolset get-models --rvc model.pth
```

### Environment variables (`.env`)

Optional project-local configuration via a `.env` file at the repo root
(`python-dotenv`, git-ignored). Copy `.env.example`, fill it in, and every
entry point (CLI, `web`, `ui`) loads it at startup without overriding
variables you already exported:

```powershell
Copy-Item .env.example .env
notepad .env
```

Supported variables:

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Hugging Face token for gated models (pyannote diarization). No more `--token` / `HF_TOKEN` juggling. |
| `AI_TOOLSET_CUDA_RUNTIME` | Override the CUDA/cuDNN `bin` folder (see "How it works"). |
| `AI_TOOLSET_CACHE` | Directory for pre-downloaded model caches (default `~/.cache`). |

## Voice conversion (RVC) and diarization

Two heavy, mutually-exclusive extras. RVC needs `rvc-python` + `fairseq`
(vendored, no C++ SDK required); diarization needs the modern pyannote stack.
**They cannot share one environment** (fairseq -> omegaconf<2.1 vs
pyannote -> omegaconf>=2.1), so they are declared conflicting and you install
one profile at a time:

```powershell
uv sync --extra rvc          # OR
uv sync --extra diarize
```

- `voice-convert` - RVC inference with `rvc-python` (needs a `.pth` model +
  optional index file from your own RVC training).
- `diarize` - pyannote speaker diarization -> RTTM; requires a Hugging Face
  token with the gated `pyannote/speaker-diarization-3.1` model accepted.

```powershell
uv run python -m ai_toolset voice-convert in.wav out.wav --model model.pth --index model.index
uv run python -m ai_toolset diarize meeting.wav --token hf_... --out meeting.rttm
```

## Mic capture, live transcription, and benchmarks

```powershell
uv sync --extra audio --extra stt
uv run python -m ai_toolset record-audio clip.wav --seconds 10     # or Enter to stop
uv run python -m ai_toolset live-transcribe --chunk 5 --model base
uv run python -m ai_toolset benchmark --audio sample.wav --engines faster whisper --gpus 0,1
uv run python -m ai_toolset benchmark --image frame.png --gpus 0,1  # YOLO latency
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/get_cuda_runtime.ps1` | Download + extract CUDA/cuDNN DLLs into `cuda_runtime/bin`, install `sitecustomize.py` into the venv. Idempotent. With `-Framework pytorch` or `-Framework fasterwhisper`: driver-only check, no downloads. |
| `scripts/verify_cuda.ps1` | Report GPU/driver status, runtime DLL count, real TF GPU detection (`-RunTensorFlowCheck`), and torch GPU detection (`-CheckTorch`). |

## Python package

Everything is importable as `ai_toolset` from the repo root:

```python
from ai_toolset import cuda, screen, video, dataset, images
```

A CLI wraps the common operations:

```powershell
uv run python -m ai_toolset --help
uv run python -m ai_toolset verify-cuda
uv run python -m ai_toolset select-gpus --gpus 0,1      # both GPUs
uv run python -m ai_toolset select-gpus --gpus 1        # only the second
uv run python -m ai_toolset extract-frames gameplay.mp4 frames --count 200
uv run python -m ai_toolset frames-to-video frames out.mp4 --fps 30
uv run python -m ai_toolset make-synthetic --classes "npc:50,50,220;house:43,90,139"
uv run python -m ai_toolset xml-to-csv labels/ labels.csv
uv run python -m ai_toolset select-region
uv run python -m ai_toolset capture-loop --out-dir frames --square
uv run python -m ai_toolset audio-rvc CLEAN/ RVC_DATA/ --speaker 0
uv run python -m ai_toolset transcribe audio.wav --engine faster --model base --gpus 0
uv run python -m ai_toolset tts "Hello from the toolset." out.wav --speaker ref.wav
uv run python -m ai_toolset detect img.png --gpus 0
uv run python -m ai_toolset train data.yaml --gpus 0
uv run python -m ai_toolset record-screen out.mp4 --select-region
uv run python -m ai_toolset webcam-capture --camera 0
uv run python -m ai_toolset augment data/images out/ --per-image 5
uv run python -m ai_toolset ocr screenshot.png
uv run python -m ai_toolset record-audio clip.wav --seconds 10
uv run python -m ai_toolset live-transcribe --chunk 5
uv run python -m ai_toolset tts-batch lines.txt --out-dir wavs --metadata metadata.csv
uv run python -m ai_toolset narrate lines.txt
uv run python -m ai_toolset benchmark --image frame.png --gpus 0,1
uv run python -m ai_toolset voice-convert in.wav out.wav --model model.pth   # uv sync --extra rvc
uv run python -m ai_toolset diarize in.wav --token hf_... --out out.rttm     # uv sync --extra diarize
uv run python -m ai_toolset mediapipe --solution pose                       # uv sync --extra mediapipe
uv run python -m ai_toolset web                                             # uv sync --extra web
uv run python -m ai_toolset ui                                              # uv sync --extra ui
uv run python -m ai_toolset get-models --status                             # model status/downloads
```

### GPU selection

On multi-GPU machines (e.g. GTX 1060 + GTX 1070) you can use both GPUs or
pin training to one. The selection must happen **before** `tensorflow`/`torch`
are imported, so it is applied via `CUDA_VISIBLE_DEVICES`:

```python
from ai_toolset import cuda

cuda.set_visible_gpus([0, 1])  # both GPUs
# cuda.set_visible_gpus([1])    # only the second GPU
# cuda.set_visible_gpus()       # all GPUs

cuda.configure_tf_gpus()  # after import: memory growth + visible set
```

Or use the CLI (interactive when multiple GPUs and `--gpus` omitted):

```powershell
uv run python -m ai_toolset select-gpus --gpus 0,1
uv run python examples/select_gpu.py --gpus 0,1 --framework auto
```

`detect_gpus()` lists `{index, name, driver, vram}` per physical GPU;
`verify_torch_gpu()` reports the CUDA devices PyTorch sees.

### Modules

- `cuda` - `detect_gpus()`, `detect_gpu()`, `matrix_entry(tf_version)`,
  `set_visible_gpus()`, `configure_tf_gpus()`, `configure_memory_growth()`,
  `verify_tf_gpu()`, `verify_torch_gpu()`
- `screen` - `capture(region)`, `capture_square()`, `select_region()`,
  `stream_frames()`
- `video` - `extract_frames()`, `frames_to_video()`, `record_screen()`,
  `webcam_capture()`
- `images` - `pad_to_square()`, `pad_images_in_dir()`, `split_quadrants()`,
  `split_image_dataset()`, `xml_to_csv()`, `augment_dir()`
- `dataset` - `generate_synthetic()` - writes images + keras-retinanet CSV
  labels for pipeline smoke-testing on the GPU
- `audio` - `probe_dir()`, `resample_dir()`, `split_on_silence()`,
  `make_rvc_dataset()` - voice-cloning dataset prep (requires
  `uv sync --extra voice`); plus `list_audio_devices()`, `record_mic()`
- `speech` - `transcribe_whisper()`, `transcribe_faster()`,
  `transcribe_live()`, `synthesize_tts()`, `synthesize_lines()`,
  `narrate()`, `play_audio()`
- `detect` - `detect_image()`, `detect_frame()`, `detect_stream()`,
  `draw_detections()`, `annotate()` (requires `uv sync --extra vision`)
- `train` - `train_yolo()`, `best_weights()`
- `ocr` - `ocr_image()`, `ocr_frame()`, `ocr_screen()` (requires
  `uv sync --extra ocr`)
- `voice` - `convert_voice()` RVC inference (requires `uv sync --extra rvc`)
- `diarize` - `diarize()` pyannote speaker diarization (requires
  `uv sync --extra diarize`)
- `benchmark` - `benchmark_stt()`, `benchmark_yolo()`, `print_table()`
- `mp` - `available()`, `process_frame()`, `annotate()`, `overlay()`,
  `webcam_stream()` (requires `uv sync --extra mediapipe`)
- `webapp` - `create_app()` FastAPI dashboard + `main()` uvicorn launcher
  (requires `uv sync --extra web`)
- `models` - `ensure_yolo()`, `ensure_whisper()`, `ensure_tts()`,
  `ensure_diarize()`, `ensure_rvc()`, `summarize()` - model download/status
- `streamlit_app` - Streamlit quick-UI entry point (requires `uv sync --extra ui`)

See `examples/verify_gpu.py`, `examples/make_synthetic_dataset.py`,
`examples/select_gpu.py`, and `examples/prepare_voice_dataset.py`.

## How it works

1. `uv sync` installs TensorFlow 2.10 (GPU wheels) into `.venv`.
2. `get_cuda_runtime.ps1` reads the mapping table, downloads NVIDIA's own
   redistributable CUDA runtime wheels (`nvidia-*-cu11`) plus the cuDNN
   archive from NVIDIA's redist server, extracts the DLLs into
   `cuda_runtime/bin`, and copies `sitecustomize.py` into the venv's
   site-packages.
3. At every interpreter start, `sitecustomize.py` prepends
   `cuda_runtime/bin` to `PATH`. TensorFlow finds `cudart64_110.dll` /
   `cudnn64_8.dll` there and loads the GPU plugin.

Set `AI_TOOLSET_CUDA_RUNTIME` to point at a `bin` folder to bypass the
auto-discovery (e.g. an already-populated shared runtime).
