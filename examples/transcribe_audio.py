"""Transcribe audio with openai-whisper or faster-whisper (ai-toolset helpers).

Usage:
  uv run python examples/transcribe_audio.py speech.wav --engine faster
  uv run python examples/transcribe_audio.py speech.wav --engine whisper --model small --gpus 0
"""

import argparse

from ai_toolset import speech


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="audio file to transcribe")
    parser.add_argument("--engine", choices=["whisper", "faster"], default="faster")
    parser.add_argument("--model", default="base",
                        help="whisper size or HF name (tiny/base/small/medium/large-v3)")
    parser.add_argument("--language", help="audio language code (default: auto)")
    parser.add_argument("--gpus", help="comma-separated GPU indices, e.g. 0,1")
    args = parser.parse_args()

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None

    if args.engine == "whisper":
        result = speech.transcribe_whisper(args.audio, model=args.model,
                                           language=args.language, gpus=gpus)
        print(result["text"].strip())
    else:
        segments, info = speech.transcribe_faster(args.audio, model=args.model,
                                                  language=args.language, gpus=gpus)
        print(f"detected language: {info.language} "
              f"(p={info.language_probability:.2f})")
        for line in speech.segment_lines(segments):
            print(line)


if __name__ == "__main__":
    main()
