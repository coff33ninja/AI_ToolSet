"""Synthesize speech with Coqui TTS - XTTS v2 voice cloning or a plain model.

Usage:
  uv run python examples/synthesize_speech.py "Hello from the toolkit" out.wav \
      --speaker reference.wav --language en
  uv run python examples/synthesize_speech.py "Hello" out.wav \
      --model tts_models/en/ljspeech/tacotron2-DDC   # no reference needed
"""

import argparse

from ai_toolset import speech


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="text to synthesize")
    parser.add_argument("out", help="output wav path")
    parser.add_argument("--model", default="tts_models/multilingual/multi-dataset/xtts_v2")
    parser.add_argument("--speaker", help="reference wav for voice cloning (XTTS requires it)")
    parser.add_argument("--language", default="en")
    parser.add_argument("--gpus", help="comma-separated GPU indices, e.g. 0,1")
    args = parser.parse_args()

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    speech.synthesize_tts(
        args.text,
        args.out,
        model_name=args.model,
        speaker_wav=args.speaker,
        language=args.language,
        gpus=gpus,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
