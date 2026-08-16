"""OCR via Windows.Media.Ocr (winocr).

Windows 10+ ships OCR language packs; winocr wraps the built-in engine, so
there are no model downloads and no torch dependency. The engine accepts a
PIL image or a BGR numpy frame (winocr handles the conversion).

Requires the `ocr` extra:  uv sync --extra ocr
"""

import cv2


def ocr_image(path, language="en"):
    """OCR an image file. Returns (text, lines) where lines is a list of
    {text, line_index, word_count} dicts."""
    img = cv2.imread(path)
    if img is None:
        raise OSError(f"Could not read image: {path}")
    return ocr_frame(img, language)


def ocr_frame(frame, language="en"):
    """OCR a BGR numpy frame. Returns (text, lines)."""
    import winocr

    result = winocr.recognize_cv2(frame, language)
    lines = []
    for line in getattr(result, "lines", []):
        lines.append({
            "text": getattr(line, "text", ""),
            "line_index": getattr(line, "line_index", 0),
            "word_count": getattr(line, "word_count", 0),
        })
    return getattr(result, "text", "").strip(), lines


def ocr_screen(region=None, language="en"):
    """OCR a screen region (or full desktop). Returns (text, lines)."""
    from ai_toolset.screen import capture

    return ocr_frame(capture(region), language)
