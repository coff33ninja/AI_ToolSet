"""Speech helpers that need no model weights or GPU."""

import os

from ai_toolset.speech import (
    _coqui_numpy_compat,
    _tts_cache_dir,
    segment_lines,
    tts_model_available,
)


def test_coqui_numpy_compat_runs():
    _coqui_numpy_compat()
    import numpy as np

    assert hasattr(np, "dtypes"), "numpy shim did not apply"


def test_tts_model_available_false_for_bogus():
    assert tts_model_available("tts_models/nope/nothing") is False


def test_tts_cache_dir_shape():
    path = _tts_cache_dir("tts_models/en/ljspeech/tacotron2-DDC")
    assert "tts_models--en--ljspeech--tacotron2-DDC" in path


def test_segment_lines_format():
    class FakeSeg:
        start = 1.5
        end = 2.5
        text = "  hello  "

    lines = segment_lines([FakeSeg()])
    assert lines[0] == "1.50s -> 2.50s  hello"


def test_transcribe_faster_rejects_missing_file(tmp_path):
    from ai_toolset.speech import transcribe_faster

    missing = os.path.join(tmp_path, "missing.wav")
    try:
        transcribe_faster(missing, model="tiny")
    except Exception as exc:  # noqa: BLE001 - any error beats a silent hang
        assert type(exc).__name__ in ("ValueError", "OSError", "FileNotFoundError", "RuntimeError")
        return
    raise AssertionError("expected transcribe_faster to fail on a missing file")
