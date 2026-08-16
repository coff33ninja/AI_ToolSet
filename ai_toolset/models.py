"""Pre-download model weights for the toolkit's backends.

YOLO, whisper, and TTS models auto-download on first use, so this is mainly a
convenience to fetch them up front (and to run the gated/assisted downloads
that cannot happen automatically: pyannote diarization needs a HF token, RVC
needs a user-trained .pth model).
"""

import os
import sys


def _cache_dir(name):
    base = os.environ.get("AI_TOOLSET_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", name
    )
    os.makedirs(base, exist_ok=True)
    return base


def ensure_yolo(weights="yolov8n.pt", gpus=None):
    """Download a YOLO weights file via ultralytics. Returns the path."""
    from ai_toolset.detect import _load

    _load(weights, gpus=gpus)
    return weights


def ensure_whisper(model="base", engine="faster", gpus=None):
    """Download a whisper model. engine='faster' -> CTranslate2 HF cache,
    engine='whisper' -> OpenAI ~/.cache/whisper. Returns a description."""
    if engine == "faster":
        from ai_toolset.cuda import set_visible_gpus

        set_visible_gpus(gpus)
        from faster_whisper import WhisperModel

        WhisperModel(model, device="auto")
        return f"faster-whisper '{model}' -> Hugging Face cache"
    from ai_toolset import speech

    speech.download_whisper(model)
    return f"openai-whisper '{model}' -> ~/.cache/whisper"


def ensure_tts(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpus=None):
    """Download a Coqui TTS model into %LOCALAPPDATA%/tts (or ~/.local/share/tts)."""
    from ai_toolset import speech

    speech.download_tts(model_name, gpus=gpus)
    return f"coqui '{model_name}' -> %LOCALAPPDATA%\\tts"


def ensure_diarize(token=None, gpus=None):
    """Download the gated pyannote diarization pipeline. Needs an HF token with
    access to pyannote/speaker-diarization-3.1 (+ pyannote/segmentation-3.0).
    Accepts the model on your HF account first: huggingface.co/pyannote/..."""
    from ai_toolset.diarize import pipeline_kwargs
    from ai_toolset.env import load_env

    load_env()
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Diarization needs a Hugging Face token with access to "
            "pyannote/speaker-diarization-3.1. Get one at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1 and pass "
            "--token or set HF_TOKEN."
        )
    kwargs = pipeline_kwargs(token=token, gpus=gpus)
    from pyannote.audio import Pipeline

    Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", **kwargs)
    return "pyannote/speaker-diarization-3.1 -> Hugging Face cache"


def ensure_rvc(model_path=None):
    """RVC models are user-trained and cannot be downloaded. Validates a .pth
    and points at where to put it."""
    if not model_path:
        raise RuntimeError(
            "RVC uses a user-trained voice model (.pth). Put one under "
            "rvc_models/<name>/<name>.pth (+ optional .index) and pass that path."
        )
    if not os.path.isfile(model_path):
        raise OSError(f"Model file not found: {model_path}")
    return model_path


def summarize():
    """Print a status report of the toolkit's model requirements."""
    rows = [
        ("YOLO", "ultralytics (auto)", "uv run python -m ai_toolset get-models --yolo"),
        ("Whisper STT", "faster-whisper (auto)", "uv run python -m ai_toolset get-models --whisper base"),
        ("TTS", "coqui (auto)", "uv run python -m ai_toolset get-models --tts"),
        ("Diarization", "gated HF model (token)", "uv run python -m ai_toolset get-models --diarize"),
        ("RVC", "user-trained .pth", "uv run python -m ai_toolset get-models --rvc <path>"),
    ]
    width = max(len(r[0]) for r in rows)
    for name, kind, cmd in rows:
        print(f"{name:>{width}}  {kind:<32} {cmd}")
    return 0


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m ai_toolset get-models",
        description="Pre-download model weights for the toolkit backends.",
    )
    parser.add_argument("--yolo", nargs="?", const="yolov8n.pt", default=None,
                        metavar="WEIGHTS", help="download YOLO weights (default yolov8n.pt)")
    parser.add_argument("--whisper", nargs="?", const="base", default=None,
                        metavar="SIZE", help="download a whisper model (default base)")
    parser.add_argument("--engine", choices=["faster", "whisper"], default="faster",
                        help="whisper engine (default faster)")
    parser.add_argument("--tts", nargs="?", const="tts_models/en/ljspeech/tacotron2-DDC",
                        default=None, metavar="MODEL",
                        help="download a coqui TTS model")
    parser.add_argument("--diarize", nargs="?", const=True, default=None,
                        metavar="TOKEN", help="HF token for pyannote diarization")
    parser.add_argument("--rvc", metavar="MODEL_PATH",
                        help="validate a user-trained RVC .pth model path")
    parser.add_argument("--status", action="store_true", help="print model status table")
    parser.add_argument("--gpus", help="comma-separated physical GPU indices")
    args = parser.parse_args(argv)

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None

    if args.status or not any(
        (args.yolo, args.whisper, args.tts, args.diarize, args.rvc)
    ):
        return summarize()

    try:
        if args.yolo:
            print(f"OK {ensure_yolo(args.yolo, gpus=gpus)}")
        if args.whisper:
            print(f"OK {ensure_whisper(args.whisper, engine=args.engine, gpus=gpus)}")
        if args.tts:
            print(f"OK {ensure_tts(args.tts, gpus=gpus)}")
        if args.diarize:
            token = args.diarize if isinstance(args.diarize, str) else None
            print(f"OK {ensure_diarize(token, gpus=gpus)}")
        if args.rvc:
            print(f"OK {ensure_rvc(args.rvc)}")
    except RuntimeError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
