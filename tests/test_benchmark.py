"""Benchmark output formatting (no model runs)."""

from ai_toolset.benchmark import print_table


def test_print_table_yolo_rows(capsys):
    rows = [
        {"gpus": (0,), "model": "yolov8n", "mean_ms": 12.3, "min_ms": 9.1},
        {"gpus": (1,), "model": "yolov8n", "mean_ms": 14.8, "min_ms": 11.0},
    ]
    print_table(rows)
    out = capsys.readouterr().out
    assert "yolov8n" in out
    assert "mean" in out


def test_print_table_stt_rows(capsys):
    rows = [
        {"gpus": (0,), "engine": "faster", "model": "base", "mean_s": 2.1, "min_s": 1.9},
    ]
    print_table(rows)
    out = capsys.readouterr().out
    assert "faster" in out
