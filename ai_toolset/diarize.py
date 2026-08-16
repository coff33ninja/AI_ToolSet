"""Speaker diarization via pyannote.audio.

Requires a Hugging Face token with access granted to the gated model
`pyannote/speaker-diarization-3.1` (accept the license on the model page,
then set --token or HF_TOKEN). Falls back to a clear error without it.

Heavy graph (speechbrain, onnxruntime, pytorch-lightning, ...) - install with:

    uv sync --extra diarize
"""

import os


def pipeline_kwargs(token=None, gpus=None):
    """Resolve the HF token + GPU selection for Pipeline.from_pretrained.

    Raises a clear RuntimeError when no token is configured, so callers get
    the same message whether they are downloading or running diarization.
    """
    from ai_toolset.cuda import set_visible_gpus
    from ai_toolset.env import load_env

    load_env()
    token = token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "pyannote diarization needs a Hugging Face token with access to "
            "pyannote/speaker-diarization-3.1. Get one at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1 and pass "
            "--token or set HF_TOKEN.")
    set_visible_gpus(gpus)
    return {"use_auth_token": token}


def diarize(audio_path, out_path=None, token=None, min_speakers=None,
            max_speakers=None, gpus=None):
    """Diarize an audio file. Writes an RTTM file when out_path is given.

    Returns a list of segments: {start, end, speaker, label}. The RTTM format
    groups turns per speaker; labels are normalized to SPEAKER_00, 01, ...
    """
    kwargs = pipeline_kwargs(token=token, gpus=gpus)

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", **kwargs)
    kwargs = {}
    if min_speakers:
        kwargs["min_speakers"] = min_speakers
    if max_speakers:
        kwargs["max_speakers"] = max_speakers
    diarization = pipeline(audio_path, **kwargs)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for seg in segments:
                dur = seg["end"] - seg["start"]
                f.write(f"SPEAKER {os.path.basename(audio_path)} 1 "
                        f"{seg['start']:.3f} {dur:.3f} <NA> <NA> "
                        f"{seg['speaker']} <NA> <NA>\n")
    return segments
