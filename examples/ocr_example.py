"""OCR an image file or a live screen region with Windows.Media.Ocr.

Usage:
  uv run python examples/ocr_example.py screenshot.png
  uv run python examples/ocr_example.py --screen
"""

import argparse

from ai_toolset import ocr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="image file to OCR")
    parser.add_argument("--screen", action="store_true",
                        help="OCR the full desktop instead of a file")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    if args.screen:
        text, lines = ocr.ocr_screen(language=args.language)
    elif args.image:
        text, lines = ocr.ocr_image(args.image, language=args.language)
    else:
        parser.error("pass an image path or --screen")

    print(text)
    for line in lines:
        print(f"  [{line['line_index']}] {line['text']}")


if __name__ == "__main__":
    main()
