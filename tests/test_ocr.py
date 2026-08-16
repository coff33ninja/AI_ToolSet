"""OCR via Windows.Media.Ocr on a synthetic text image.

Skips when the `ocr` extra is missing or no English OCR language pack is
installed (Windows component, not installable via pip).
"""

import cv2
import numpy as np
import pytest

from ai_toolset.ocr import ocr_image


def _has_en_ocr():
    try:
        import winocr

        return any(lang.tag.lower().startswith("en") for lang in winocr.languages())
    except Exception:  # noqa: BLE001 - import or enumeration failure
        return False


requires_en_ocr = pytest.mark.skipif(
    not _has_en_ocr(), reason="winocr missing or no English OCR language pack"
)


@requires_en_ocr
def test_ocr_image_reads_text(tmp_path):
    path = str(tmp_path / "text.png")
    img = np.zeros((160, 640, 3), dtype=np.uint8)
    cv2.putText(
        img, "Hello AI Toolset", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
    )
    cv2.imwrite(path, img)
    text, lines = ocr_image(path)
    assert text, "OCR returned empty text"
    assert lines


@requires_en_ocr
def test_ocr_image_missing_file(tmp_path):
    with pytest.raises(OSError):
        ocr_image(str(tmp_path / "nope.png"))
