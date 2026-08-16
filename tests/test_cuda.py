"""CUDA helpers: TF/CUDA matrix, GPU visibility env var."""

import os

import pytest

from ai_toolset.cuda import detect_gpus, matrix_entry, set_visible_gpus

TF_MATRIX_ENTRIES = ("cuda", "cudnn", "driver_min")


def test_matrix_entry_shape():
    for tf_version in ("2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "2.10"):
        entry = matrix_entry(tf_version)
        assert set(TF_MATRIX_ENTRIES).issubset(entry)


def test_matrix_entry_unknown_raises():
    with pytest.raises(KeyError):
        matrix_entry("99.99")


def test_set_visible_gpus_all(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    for value in (None, "all", []):
        assert set_visible_gpus(value) is None
        assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_set_visible_gpus_select(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert set_visible_gpus([0, 2]) == [0, 2]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,2"


def test_detect_gpus_returns_list():
    gpus = detect_gpus()
    assert isinstance(gpus, list)
    for gpu in gpus:
        assert {"index", "name", "driver", "vram"}.issubset(gpu)
