"""Prepare a voice-cloning dataset (RVC format) using the ai-toolset audio helpers.

Pipeline: resample every recording to 48 kHz mono WAV, then split on silence
into segments named <speaker>_<index>.wav as the RVC webui expects.

Usage (from the repo root, venv-local):
    uv sync --extra voice
    uv run python examples/prepare_voice_dataset.py RAW_AUDIO_DIR OUT_DIR --speaker 0
"""

import argparse
import os

from ai_toolset import audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_dir", help="folder of raw recordings (wav/mp3/flac/...)")
    parser.add_argument("out_dir", help="where the RVC-ready WAV segments go")
    parser.add_argument("--speaker", type=int, default=0,
                        help="speaker id used as the <id>_<index> filename prefix")
    parser.add_argument("--sr", type=int, default=48000, help="target sample rate")
    parser.add_argument("--max-sec", type=float, default=60, help="max segment length")
    parser.add_argument("--min-silence", type=float, default=0.5,
                        help="silence gap (s) that separates segments")
    args = parser.parse_args()

    resampled = os.path.join(args.out_dir, "resampled")
    print(f"Step 1/2 - resample to {args.sr // 1000} kHz mono WAV ...")
    n = audio.resample_dir(args.src_dir, resampled, sr=args.sr)
    print(f"  resampled {n} files into {resampled}")

    print("Step 2/2 - split on silence into RVC segments ...")
    m = audio.make_rvc_dataset(
        resampled, args.out_dir, speaker_id=args.speaker,
        max_sec=args.max_sec, sr=args.sr, min_silence=args.min_silence)
    print(f"  wrote {m} segments to {args.out_dir} (speaker {args.speaker})")


if __name__ == "__main__":
    main()
