"""Speech-to-text (Whisper, faster-whisper) and text-to-speech (Coqui XTTS).

Designed for the project-local workflow:

  uv sync --extra stt   # openai-whisper + faster-whisper
  uv sync --extra tts   # coqui-tts (XTTS v2)

Backends:
  * openai-whisper runs on the cu118 torch from the voice extra. It honors
    CUDA_VISIBLE_DEVICES, so pass gpus= to restrict which physical GPU is used.
  * faster-whisper runs on CTranslate2. Its PyPI wheels support GPU directly;
    the only system requirement is CUDA 12.x + cuDNN 8-for-CUDA-12 reachable
    at runtime (scripts/get_cuda_runtime.ps1 -Framework fasterwhisper
    provisions those project-locally). Without it the engine transparently
    falls back to CPU with int8 quantization.
  * Coqui TTS (XTTS v2 by default) does voice-cloning synthesis on the same
    torch stack; pass a reference speaker wav for cloning.
"""

from ai_toolset.cuda import set_visible_gpus


def _apply_gpus(gpus):
    """Set CUDA_VISIBLE_DEVICES before any framework import."""
    if gpus is not None:
        set_visible_gpus(gpus)


def transcribe_whisper(audio_path, model="base", language=None, device=None,
                       task="transcribe", fp16=True, verbose=False, gpus=None):
    """Transcribe audio with OpenAI Whisper (PyTorch backend).

    Returns the whisper result dict; the transcript is result["text"]. With
    task="translate" non-English audio is translated to English instead.
    device defaults to cuda when available. fp16 is only used on cuda.
    """
    _apply_gpus(gpus)
    import torch
    import whisper

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaded = whisper.load_model(model, device=device)
    return loaded.transcribe(
        audio_path, language=language, task=task,
        fp16=(bool(fp16) and device == "cuda"), verbose=verbose,
    )


def transcribe_faster(audio_path, model="base", language=None, device="auto",
                      compute_type="auto", beam_size=5, vad_filter=True,
                      condition_on_previous_text=True, gpus=None):
    """Transcribe audio with faster-whisper (CTranslate2 backend).

    device "auto" selects cuda when CTranslate2 sees a usable GPU, else cpu.
    compute_type "auto" = float16 on cuda, int8 on cpu. Returns
    (segments, info); segments is a fully materialized list so transcription
    runs before returning.
    """
    _apply_gpus(gpus)
    from faster_whisper import WhisperModel

    if device == "auto":
        import ctranslate2
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if compute_type == "auto":
        import ctranslate2
        supported = set(ctranslate2.get_supported_compute_types(device))
        if "float16" in supported:
            compute_type = "float16"
        elif "int8_float16" in supported:
            compute_type = "int8_float16"
        elif device == "cuda":
            # Pascal (GTX 10-series) has no efficient fp16 in CTranslate2.
            compute_type = "float32"
        else:
            compute_type = "int8"
    loaded = WhisperModel(model, device=device, compute_type=compute_type)
    segments, info = loaded.transcribe(
        audio_path, language=language, beam_size=beam_size,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
    )
    return list(segments), info


def segment_lines(segments):
    """Format faster-whisper segments as 'SS.SSs -> SS.SSs  text' lines."""
    return [f"{s.start:.2f}s -> {s.end:.2f}s  {s.text.strip()}" for s in segments]


def synthesize_tts(text, out_path, model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                   speaker_wav=None, language="en", device=None, gpus=None):
    """Synthesize speech with Coqui TTS.

    XTTS v2 (the default) is a voice-cloning model and REQUIRES speaker_wav (a
    reference recording, >= 6 s, ideally clean). Pass a single-speaker model
    such as "tts_models/en/ljspeech/tacotron2-DDC" to synthesize without a
    reference clip. language is only used for multilingual models. Models are
    downloaded once from the Hugging Face Hub into the local cache. Returns
    out_path.
    """
    _apply_gpus(gpus)
    _coqui_numpy_compat()
    import torch
    from TTS.api import TTS

    if device is None:
        use_gpu = bool(torch.cuda.is_available())
    else:
        use_gpu = str(device).lower().startswith(("cuda", "gpu"))
    tts = TTS(model_name, gpu=use_gpu)
    kwargs = {}
    if speaker_wav:
        kwargs["speaker_wav"] = speaker_wav
    if tts.is_multi_lingual:
        kwargs["language"] = language
    tts.tts_to_file(text, file_path=out_path, **kwargs)
    return out_path


def _coqui_numpy_compat():
    """Shim coqui-tts's numpy>=1.24 requirement onto the TF-pinned 1.23.5.

    coqui-tts 0.24.x references np.dtypes.Float64DType in TTS/__init__.py (a
    torch.serialization safe-global registration), but np.dtypes only exists
    in numpy>=1.24 while TensorFlow 2.10 pins the graph to numpy 1.23.5. The
    only usage is that import-time registration, so a minimal attribute shim
    satisfies it without touching the pinned numpy.
    """
    import numpy as np

    if not hasattr(np, "dtypes"):
        np.dtypes = type(
            "np_dtypes_compat",
            (),
            {"Float64DType": np.dtype(np.float64).__class__},
        )


def transcribe_live(duration=60, chunk=5, model="base", language=None,
                    gpus=None, out_dir="live_segments"):
    """Stream-transcribe the microphone with faster-whisper.

    Records `chunk`-second windows and transcribes each one as it lands,
    printing the transcript as it goes. duration<=0 runs until Ctrl+C. Each
    window is also written to out_dir as live_seg_<n>.wav (16 kHz mono).
    Requires the `audio` + `stt` extras:
        uv sync --extra audio --extra stt
    Returns the list of (wav_path, transcript).
    """
    import os
    import tempfile
    import time

    from ai_toolset.audio import record_mic

    os.makedirs(out_dir, exist_ok=True)
    loaded = None
    results = []
    n = 0
    try:
        while True:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = tmp.name
            path, seconds = record_mic(path, duration=chunk, sr=16000)
            if seconds < 0.5:
                os.remove(tmp.name)
                if duration <= 0:
                    time.sleep(0.5)
                continue
            final_path = os.path.join(out_dir, f"live_seg_{n:04d}.wav")
            os.replace(path, final_path)
            if loaded is None:
                _apply_gpus(gpus)
                from faster_whisper import WhisperModel
                loaded = WhisperModel(model, device="cpu", compute_type="int8")
            segments, info = loaded.transcribe(final_path, language=language)
            text = " ".join(s.text.strip() for s in segments).strip()
            print(f"[{info.language} p={info.language_probability:.2f}] {text or '(silence)'}")
            results.append((final_path, text))
            n += 1
            if duration > 0 and n * chunk >= duration:
                break
    except KeyboardInterrupt:
        pass
    return results


def synthesize_lines(lines, out_dir="tts_output", model_name=
                     "tts_models/multilingual/multi-dataset/xtts_v2",
                     speaker_wav=None, language="en", gpus=None, prefix="line",
                     metadata_csv=None):
    """Synthesize one wav per text line (TTS batch / dataset builder).

    Skips empty lines. Writes <prefix>_<i:04d>.wav per line. When metadata_csv
    is given (e.g. "metadata.csv"), a Coqui-format "path|text" file is written
    alongside for fine-tuning datasets. Returns the list of written wav paths.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    written = []
    meta = []
    for i, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue
        out_path = os.path.join(out_dir, f"{prefix}_{i:04d}.wav")
        synthesize_tts(text, out_path, model_name=model_name,
                       speaker_wav=speaker_wav, language=language, gpus=gpus)
        written.append(out_path)
        if metadata_csv:
            meta.append(f"{os.path.basename(out_path)}|{text}")
    if metadata_csv:
        meta_path = metadata_csv if os.path.isabs(metadata_csv) else \
            os.path.join(out_dir, metadata_csv)
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write("\n".join(meta) + "\n")
    return written


def play_audio(path):
    """Play a WAV file through the default audio device (sounddevice)."""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    _play_float32(data, sr)


def _play_float32(data, sr):
    import sounddevice as sd

    sd.play(data, sr)
    sd.wait()


def narrate(lines, model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            speaker_wav=None, language="en", gpus=None, out_dir=None):
    """Synthesize each line and play it out loud in sequence.

    out_dir optionally keeps the generated wavs (default: skip saving).
    Requires the `tts` + `audio` extras. Returns the list of wav paths played.
    """
    import os
    import tempfile

    paths = []
    for i, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            wav = os.path.join(out_dir, f"narrate_{i:04d}.wav")
        else:
            fd, wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
        synthesize_tts(text, wav, model_name=model_name,
                       speaker_wav=speaker_wav, language=language, gpus=gpus)
        play_audio(wav)
        paths.append(wav)
    return paths
