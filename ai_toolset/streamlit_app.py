"""Streamlit quick-UI for the ai-toolset (STT, TTS, detection, OCR).

Run with:  uv run python -m ai_toolset ui

The heavy backends are loaded once and cached per session; the sidebar GPU
picker flows into every model call. Falls back to CPU gracefully.
"""

import os
import tempfile

import streamlit as st

from ai_toolset.env import load_env

load_env()

st.set_page_config(page_title="AI ToolSet Quick UI", page_icon=":material/auto_awesome:",
                   layout="wide")


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
st.sidebar.caption("Select-GPUs = physical indices; CUDA_VISIBLE_DEVICES is set "
                   "per call before any framework import.")

tab_stt, tab_tts, tab_detect, tab_ocr, tab_models = st.tabs(
    ["Speech-to-text", "Text-to-speech", "Detection", "OCR", "Model downloads"])


@st.cache_resource(show_spinner="loading whisper...")
def load_whisper(model):
    from faster_whisper import WhisperModel

    return WhisperModel(model, device="auto")


@st.cache_resource(show_spinner="loading YOLO...")
def load_yolo(weights):
    from ai_toolset.detect import _load

    return _load(weights)


@st.cache_resource(show_spinner="loading TTS...")
def load_tts(model_name):
    from ai_toolset.speech import _coqui_numpy_compat

    _coqui_numpy_compat()
    from TTS.api import TTS

    return TTS(model_name, gpu=False)


with tab_stt:
    st.subheader("Transcribe audio")
    model = st.selectbox("Model", ["tiny", "base", "small", "medium"], index=1)
    audio = st.file_uploader("Audio file", type=["wav", "mp3", "ogg", "m4a"])
    if audio is not None and st.button("Transcribe", type="primary"):
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio.name)[1]) as tmp:
            tmp.write(audio.getvalue())
            tmp.flush()
            from ai_toolset.speech import transcribe_faster

            segments, info = transcribe_faster(tmp.name, model=model, gpus=gpus)
        st.caption(f"language {info.language} (p={info.language_probability:.2f})")
        st.code("\n".join(s.text.strip() for s in segments), language=None)

with tab_tts:
    st.subheader("Synthesize speech")
    tts_models = {
        "tts_models/en/ljspeech/tacotron2-DDC": "LJSpeech Tacotron2 (single speaker, no ref)",
        "tts_models/en/ljspeech/fast_pitch": "LJSpeech FastPitch (single speaker, no ref)",
        "tts_models/en/vctk/vits": "VCTK VITS (multi-speaker, no ref)",
        "tts_models/multilingual/multi-dataset/xtts_v2": "XTTS v2 (voice cloning, needs reference wav)",
    }
    tts_name = st.selectbox("Model", list(tts_models), format_func=lambda m: f"{m}  -  {tts_models[m]}")
    speaker = st.file_uploader("Reference wav (XTTS voice cloning)", type=["wav"])
    text = st.text_area("Text", height=80)
    if text and st.button("Synthesize", type="primary"):
        if tts_name.endswith("xtts_v2") and speaker is None:
            st.error("XTTS v2 needs a reference wav (>= 6 s) for voice cloning.")
        else:
            with st.spinner("synthesizing..."):
                load_tts(tts_name)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
                    out_path = out.name
                speaker_path = None
                if speaker is not None:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as sp:
                        sp.write(speaker.getvalue())
                        speaker_path = sp.name
                from ai_toolset.speech import synthesize_tts

                synthesize_tts(text, out_path, model_name=tts_name,
                               speaker_wav=speaker_path, gpus=gpus)
            st.audio(out_path)
            os.remove(out_path)
            if speaker_path:
                os.remove(speaker_path)

with tab_detect:
    st.subheader("YOLO detection")
    weights = st.text_input("Weights", value="yolov8n.pt")
    conf = st.slider("Confidence", 0.0, 1.0, 0.25)
    img = st.file_uploader("Image", type=["jpg", "png", "jpeg", "bmp"])
    if img is not None and st.button("Detect", type="primary"):
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(img.name)[1]) as tmp:
            tmp.write(img.getvalue())
            tmp.flush()
            import cv2

            from ai_toolset.detect import detect_frame, draw_detections

            model = load_yolo(weights)
            frame = cv2.imread(tmp.name)
            detections = detect_frame(frame, weights=weights, conf=conf,
                                      model=model, gpus=gpus)
            draw_detections(frame, detections)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                 caption=f"{len(detections)} detections")
        st.dataframe(detections, use_container_width=True)

with tab_ocr:
    st.subheader("OCR (Windows.Media.Ocr)")
    img = st.file_uploader("Image", type=["jpg", "png", "jpeg", "bmp"])
    if img is not None and st.button("Read text", type="primary"):
        from ai_toolset.ocr import ocr_image

        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(img.name)[1]) as tmp:
            tmp.write(img.getvalue())
            tmp.flush()
            text, lines = ocr_image(tmp.name)
        st.write(text or "(no text found)")
        if lines:
            st.dataframe(lines, use_container_width=True)

with tab_models:
    st.subheader("Pre-download models")
    st.caption("YOLO/whisper/TTS auto-download on first use; diarization needs an "
               "HF token and RVC needs a user-trained model. These buttons fetch "
               "everything up front.")
    from ai_toolset import models

    if st.button("YOLO yolov8n.pt", use_container_width=True):
        with st.spinner("downloading..."):
            models.ensure_yolo()
        st.success("done")
    if st.button("Whisper base (faster-whisper)", use_container_width=True):
        with st.spinner("downloading..."):
            models.ensure_whisper("base")
        st.success("done")
    if st.button("TTS tacotron2-DDC", use_container_width=True):
        with st.spinner("downloading..."):
            models.ensure_tts()
        st.success("done")
    token = st.text_input("HF token (for diarization)", type="password")
    if st.button("Diarization pipeline", use_container_width=True):
        try:
            with st.spinner("downloading (this is large)..."):
                models.ensure_diarize(token or None)
            st.success("done")
        except RuntimeError as exc:
            st.error(str(exc))
