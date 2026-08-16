"""Streamlit quick-UI for the ai-toolset (STT, TTS, detection, OCR,
benchmark, MediaPipe, audio capture, model downloads).

Run with:  uv run python -m ai_toolset ui

The heavy backends are loaded once and cached per session; the sidebar GPU
picker flows into every model call. Falls back to CPU gracefully.
"""

import contextlib
import os
import shutil
import tempfile

import streamlit as st

from ai_toolset.env import load_env

load_env()

st.set_page_config(
    page_title="AI ToolSet Quick UI", page_icon=":material/auto_awesome:", layout="wide"
)

WHISPER_SIZES = ["tiny", "base", "small", "medium", "large-v3"]

TTS_MODELS = {
    "tts_models/en/ljspeech/tacotron2-DDC": "LJSpeech Tacotron2 (single speaker, no ref)",
    "tts_models/en/ljspeech/fast_pitch": "LJSpeech FastPitch (single speaker, no ref)",
    "tts_models/en/vctk/vits": "VCTK VITS (multi-speaker, no ref)",
    "tts_models/multilingual/multi-dataset/xtts_v2": "XTTS v2 (voice cloning, needs reference wav)",
}


def _gpus():
    from ai_toolset.cuda import detect_gpus

    return detect_gpus()


def _gpu_list():
    gpus = _gpus()
    if not gpus:
        st.sidebar.info("No NVIDIA GPU detected - running on CPU.")
        return None
    options = {f"{g['index']} - {g['name']}": g["index"] for g in gpus}
    picked = st.sidebar.multiselect("GPU(s)", list(options), default=list(options)[:1])
    return [options[p] for p in picked]


gpus = _gpu_list()
st.sidebar.divider()
st.sidebar.caption(
    "Select-GPUs = physical indices; CUDA_VISIBLE_DEVICES is set "
    "per call before any framework import."
)


def _gpu_key():
    return tuple(gpus) if gpus else ()


def _write_temp(uploaded, suffix):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded.getvalue())
    return path


def _remove_paths(*paths):
    for p in paths:
        if p:
            with contextlib.suppress(OSError):
                os.remove(p)


@st.cache_resource(show_spinner="loading whisper...")
def load_whisper(model, engine, gpu_key):
    from ai_toolset.cuda import set_visible_gpus

    set_visible_gpus(gpu_key)
    if engine == "whisper":
        import torch
        import whisper

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return whisper.load_model(model, device=device)
    from ai_toolset.speech import load_faster_model

    return load_faster_model(model, gpus=gpu_key)


@st.cache_resource(show_spinner="loading YOLO...")
def load_yolo(weights):
    from ai_toolset.detect import _load

    return _load(weights)


@st.cache_resource(show_spinner="loading TTS...")
def load_tts(model_name):
    from ai_toolset.speech import ensure_tts_model

    return ensure_tts_model(model_name)


tab_stt, tab_tts, tab_detect, tab_ocr, tab_bench, tab_mp, tab_audio, tab_models = st.tabs(
    [
        "Speech-to-text",
        "Text-to-speech",
        "Detection",
        "OCR",
        "Benchmark",
        "MediaPipe",
        "Audio capture",
        "Model downloads",
    ]
)


with tab_stt:
    st.subheader("Transcribe audio")
    engine = st.radio(
        "Engine",
        ["faster", "whisper"],
        horizontal=True,
        help="faster-whisper (CTranslate2) or openai-whisper",
    )
    model = st.selectbox("Model", WHISPER_SIZES, index=1)
    language = st.text_input(
        "Language (optional)", value="", help="e.g. 'en' or 'ja'; blank = auto-detect"
    )
    task = st.selectbox(
        "Task", ["transcribe", "translate"], help="translate only applies to the whisper engine"
    )
    audio = st.file_uploader("Audio file", type=["wav", "mp3", "ogg", "m4a"])
    if audio is not None and st.button("Transcribe", type="primary"):
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio.name)[1]) as tmp:
            tmp.write(audio.getvalue())
            tmp.flush()
            lang = language or None
            if engine == "faster":
                loaded = load_whisper(model, "faster", _gpu_key())
                segments, info = loaded.transcribe(
                    tmp.name, language=lang, beam_size=5, vad_filter=True
                )
                st.caption(f"language {info.language} (p={info.language_probability:.2f})")
                st.code("\n".join(s.text.strip() for s in segments), language=None)
            else:
                from ai_toolset.speech import transcribe_whisper

                result = transcribe_whisper(
                    tmp.name, model=model, language=lang, task=task, gpus=gpus
                )
                st.code(result["text"].strip(), language=None)

with tab_tts:
    from ai_toolset.speech import synthesize_lines, synthesize_tts, tts_model_available

    st.subheader("Synthesize speech")
    tts_name = st.selectbox(
        "Model", list(TTS_MODELS), format_func=lambda m: f"{m}  -  {TTS_MODELS[m]}"
    )
    language = st.text_input("Language (XTTS v2 only)", value="en")
    speaker = st.file_uploader("Reference wav (XTTS voice cloning)", type=["wav"])
    text = st.text_area("Text", height=80)

    if not tts_model_available(tts_name):
        st.info(
            f"`{tts_name}` is not downloaded yet. Allocate it from the "
            "**Model downloads** tab (TTS dropdown)."
        )
    if text and st.button("Synthesize", type="primary"):
        if tts_name.endswith("xtts_v2") and speaker is None:
            st.error("XTTS v2 needs a reference wav (>= 6 s) for voice cloning.")
        elif not tts_model_available(tts_name):
            st.info(
                f"`{tts_name}` is not downloaded yet. Allocate it from the "
                "**Model downloads** tab (TTS dropdown)."
            )
        else:
            out_path = speaker_path = None
            try:
                fd, out_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                speaker_path = _write_temp(speaker, ".wav") if speaker is not None else None
                with st.spinner("synthesizing..."):
                    synthesize_tts(
                        text,
                        out_path,
                        model_name=tts_name,
                        speaker_wav=speaker_path,
                        language=language,
                        gpus=gpus,
                    )
                with open(out_path, "rb") as f:
                    data = f.read()
                st.audio(data, format="audio/wav")
                st.download_button(
                    "Download wav", data, file_name=os.path.basename(out_path), mime="audio/wav"
                )
            finally:
                _remove_paths(out_path, speaker_path)

    st.divider()
    st.subheader("Batch synthesize (one wav per line)")
    batch_text = st.text_area("Lines (one per line)", height=100)
    batch_prefix = st.text_input("File prefix", value="line")
    if batch_text and st.button("Synthesize batch", type="secondary"):
        if tts_name.endswith("xtts_v2") and speaker is None:
            st.error("XTTS v2 needs a reference wav (>= 6 s) for voice cloning.")
        elif not tts_model_available(tts_name):
            st.info(
                f"`{tts_name}` is not downloaded yet. Allocate it from the "
                "**Model downloads** tab (TTS dropdown)."
            )
        else:
            out_dir = tempfile.mkdtemp(prefix="tts_batch_")
            speaker_path = _write_temp(speaker, ".wav") if speaker is not None else None
            try:
                with st.spinner("synthesizing..."):
                    written = synthesize_lines(
                        batch_text.splitlines(),
                        out_dir=out_dir,
                        model_name=tts_name,
                        speaker_wav=speaker_path,
                        language=language,
                        gpus=gpus,
                        prefix=batch_prefix,
                    )
                for wav in written:
                    with open(wav, "rb") as f:
                        data = f.read()
                    st.download_button(
                        f"Download {os.path.basename(wav)}",
                        data,
                        file_name=os.path.basename(wav),
                        mime="audio/wav",
                        key=f"batch_dl_{wav}",
                    )
                st.caption(f"{len(written)} wav files synthesized")
            finally:
                _remove_paths(speaker_path)
                shutil.rmtree(out_dir, ignore_errors=True)

with tab_detect:
    st.subheader("YOLO detection")
    weights = st.text_input("Weights", value="yolov8n.pt")
    conf = st.slider("Confidence", 0.0, 1.0, 0.25)
    img = st.file_uploader("Image", type=["jpg", "png", "jpeg", "bmp"], key="detect_image")
    if img is not None and st.button("Detect", type="primary"):
        import cv2

        from ai_toolset.detect import detect_frame, draw_detections

        tmp = _write_temp(img, os.path.splitext(img.name)[1])
        try:
            model = load_yolo(weights)
            frame = cv2.imread(tmp)
            detections = detect_frame(frame, weights=weights, conf=conf, model=model, gpus=gpus)
            draw_detections(frame, detections)
            st.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"{len(detections)} detections"
            )
            ok, buf = cv2.imencode(".png", frame)
            if ok:
                st.download_button(
                    "Download annotated image",
                    buf.tobytes(),
                    file_name="annotated.png",
                    mime="image/png",
                )
            st.dataframe(detections, use_container_width=True)
        finally:
            _remove_paths(tmp)

    st.divider()
    st.subheader("YOLO on a video (sampled frames)")
    video = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"], key="detect_video")
    sample_every = st.slider("Sample every N frames", 1, 60, 15)
    if video is not None and st.button("Detect video", type="secondary"):
        import cv2

        from ai_toolset.detect import detect_frame, draw_detections

        tmp = _write_temp(video, os.path.splitext(video.name)[1])
        cap = None
        try:
            model = load_yolo(weights)
            cap = cv2.VideoCapture(tmp)
            frames_total = 0
            det_total = 0
            samples = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames_total += 1
                if frames_total % sample_every == 0:
                    detections = detect_frame(
                        frame, weights=weights, conf=conf, model=model, gpus=gpus
                    )
                    draw_detections(frame, detections)
                    det_total += len(detections)
                    samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if samples:
                cols = st.columns(3)
                for i, s in enumerate(samples[:9]):
                    with cols[i % 3]:
                        st.image(s, caption=f"frame {i * sample_every + 1}")
                st.caption(f"{len(samples)} sampled frames, {det_total} detections")
            else:
                st.info("No frames sampled (video empty or sample stride too large).")
        finally:
            if cap:
                cap.release()
            _remove_paths(tmp)

with tab_ocr:
    st.subheader("OCR (Windows.Media.Ocr)")
    ocr_lang = st.text_input("Language", value="en")
    source = st.radio("Source", ["Upload image", "Screen"], horizontal=True)
    img = st.file_uploader("Image", type=["jpg", "png", "jpeg", "bmp"], key="ocr_image")
    if st.button("Read text", type="primary"):
        from ai_toolset.ocr import ocr_image, ocr_screen

        if source == "Screen":
            text, lines = ocr_screen(region=None, language=ocr_lang or "en")
        elif img is not None:
            tmp = _write_temp(img, os.path.splitext(img.name)[1])
            try:
                text, lines = ocr_image(tmp, language=ocr_lang or "en")
            finally:
                _remove_paths(tmp)
        else:
            st.info("Upload an image or switch Source to Screen.")
            text = lines = None
        if text is not None:
            st.write(text or "(no text found)")
            if lines:
                st.dataframe(lines, use_container_width=True)

with tab_bench:
    st.subheader("Latency benchmarks")
    bench_audio = st.file_uploader(
        "Audio for STT", type=["wav", "mp3", "ogg", "m4a"], key="bench_audio"
    )
    bench_image = st.file_uploader(
        "Image for YOLO", type=["jpg", "png", "jpeg", "bmp"], key="bench_image"
    )
    stt_model = st.selectbox("STT model", WHISPER_SIZES, index=1, key="bench_stt_model")
    yolo_weights = st.text_input("YOLO weights", value="yolov8n.pt", key="bench_yolo")
    engines = st.multiselect("STT engines", ["faster", "whisper"], default=["faster"])
    iterations = st.number_input("Iterations", 1, 20, 3)
    if st.button("Run benchmark", type="primary"):
        from ai_toolset.benchmark import benchmark_stt, benchmark_yolo

        rows = []
        tmp_paths = []
        try:
            if bench_audio is not None:
                a = _write_temp(bench_audio, os.path.splitext(bench_audio.name)[1])
                tmp_paths.append(a)
                for engine in engines:
                    rows.extend(
                        benchmark_stt(
                            a, engine=engine, model=stt_model, iterations=iterations, gpus=gpus
                        )
                    )
            if bench_image is not None:
                im = _write_temp(bench_image, os.path.splitext(bench_image.name)[1])
                tmp_paths.append(im)
                rows.extend(
                    benchmark_yolo(im, weights=yolo_weights, iterations=iterations, gpus=gpus)
                )
        finally:
            _remove_paths(*tmp_paths)
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("Upload at least one audio or image file.")

with tab_mp:
    st.subheader("MediaPipe overlay")
    from ai_toolset.mp import available

    if not available():
        st.info(
            "The mediapipe extra is not installed. Install it with: `uv sync --extra mediapipe`"
        )
    else:
        import cv2
        import numpy as np

        from ai_toolset.mp import annotate, process_frame

        solution = st.selectbox("Solution", ["pose", "hands", "face", "holistic", "selfie"])
        camera = st.camera_input("Webcam", key="mp_camera")
        if camera is not None:
            arr = np.frombuffer(camera.getvalue(), np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            results = process_frame(frame, solution=solution)
            annotate(frame, solution, results)
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

with tab_audio:
    st.subheader("Record microphone")
    from ai_toolset.audio import list_audio_devices, record_mic

    try:
        devices = list_audio_devices()
        inputs = [d for d in devices if d.max_input_channels > 0]
    except Exception:  # noqa: BLE001
        inputs = []
    device_opts = (
        {f"[{i}] {d.name}": i for i, d in enumerate(inputs)} if inputs else {"default": None}
    )
    device_pick = st.selectbox("Input device", list(device_opts), key="record_device")
    rec_duration = st.slider("Duration (seconds)", 1, 60, 10)
    if st.button("Record", type="primary"):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with st.spinner("recording..."):
                path, seconds = record_mic(
                    path, rec_duration, sr=16000, device=device_opts[device_pick]
                )
            with open(path, "rb") as f:
                data = f.read()
            st.audio(data, format="audio/wav")
            st.caption(f"{seconds:.1f}s recorded")
            st.download_button(
                "Download wav", data, file_name=os.path.basename(path), mime="audio/wav"
            )
        finally:
            _remove_paths(path)

    st.divider()
    st.subheader("Live transcription (mic -> faster-whisper)")
    lt_duration = st.slider("Total seconds", 5, 300, 60, key="lt_duration")
    lt_chunk = st.slider("Window (seconds)", 2, 15, 5, key="lt_chunk")
    lt_model = st.selectbox("Model", WHISPER_SIZES, index=1, key="lt_model")
    if st.button("Start live transcription", type="secondary"):
        from ai_toolset.speech import transcribe_live

        out_dir = tempfile.mkdtemp(prefix="live_seg_")
        try:
            with st.spinner(f"recording and transcribing {lt_duration}s..."):
                results = transcribe_live(
                    duration=lt_duration,
                    chunk=lt_chunk,
                    model=lt_model,
                    language=None,
                    gpus=gpus,
                    out_dir=out_dir,
                )
            st.caption(f"{len(results)} segments")
            for wav, txt in results:
                st.write(f"**{os.path.basename(wav)}**  {txt or '(silence)'}")
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

with tab_models:
    st.subheader("Pre-download models")
    st.caption(
        "Pick a variant for each model, then download it. The TTS/detect/STT "
        "tabs use whatever has been allocated here."
    )
    from ai_toolset import models

    yolo_pick = st.selectbox(
        "YOLO weights",
        models.YOLO_WEIGHTS
        if hasattr(models, "YOLO_WEIGHTS")
        else ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
    )
    if st.button("Download YOLO weights", use_container_width=True):
        with st.spinner("downloading..."):
            models.ensure_yolo(yolo_pick)
        st.success("done")

    whisper_size = st.selectbox("Whisper size", WHISPER_SIZES, index=1, key="models_whisper_size")
    whisper_engine = st.selectbox(
        "Whisper engine", ["faster", "whisper"], key="models_whisper_engine"
    )
    if st.button("Download Whisper model", use_container_width=True):
        with st.spinner("downloading..."):
            models.ensure_whisper(whisper_size, engine=whisper_engine)
        st.success("done")

    tts_pick = st.selectbox(
        "TTS model", list(TTS_MODELS), format_func=lambda m: f"{m}  -  {TTS_MODELS[m]}"
    )
    if st.button("Download TTS model", use_container_width=True):
        with st.spinner("downloading (XTTS v2 is large)..."):
            models.ensure_tts(tts_pick)
        st.success("done")

    st.divider()
    st.caption(
        "Diarization needs a HF token (gated pyannote/speaker-diarization-3.1 "
        "model). RVC needs a user-trained model path."
    )

    def _pyannote_available():
        try:
            import pyannote.audio  # noqa: F401

            return True
        except ImportError:
            return False

    if _pyannote_available():
        token = st.text_input("HF token (for diarization)", type="password")
        if st.button("Download diarization pipeline", use_container_width=True):
            try:
                with st.spinner("downloading (this is large)..."):
                    models.ensure_diarize(token or None)
                st.success("done")
            except RuntimeError as exc:
                st.error(str(exc))
    else:
        st.info(
            "The diarize extra is not installed (pyannote conflicts with "
            "the rvc extra). Install it with: `uv sync --extra diarize`"
        )

    rvc_path = st.text_input("RVC model path (.pth)", value="")
    if rvc_path and st.button("Register RVC model", use_container_width=True):
        try:
            models.ensure_rvc(rvc_path)
            st.success("model path registered")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
