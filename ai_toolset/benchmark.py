"""Latency benchmark for the local STT and YOLO engines on each GPU.

Measures cold-warm per-inference latency for openai-whisper, faster-whisper,
and ultralytics YOLO across the GPUs selected. Uses the same project-local
engines as the CLI, so results reflect the real stack (numpy 1.23.5 + torch
2.7.1 cu118 on this machine).

Example:
    uv run python -m ai_toolset benchmark --gpus 0,1 --audio samples.wav
"""

import time


def _timed(fn, iterations):
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return timings


def benchmark_stt(audio_path, engine="faster", model="base", iterations=3, gpus=None):
    """Benchmark one STT engine on each selected GPU. Returns list of rows.

    Rows: {gpus, engine, model, mean_s, min_s}. warmup run excluded from stats.
    """
    from ai_toolset import speech

    rows = []
    for gpu in gpus or [None]:

        def run_once(engine=engine, gpu=gpu):
            if engine == "whisper":
                speech.transcribe_whisper(
                    audio_path, model=model, gpus=[gpu] if gpu is not None else None
                )
            else:
                speech.transcribe_faster(
                    audio_path, model=model, gpus=[gpu] if gpu is not None else None
                )

        run_once()  # warmup (model load dominates)
        timings = _timed(run_once, iterations)
        rows.append(
            {
                "gpus": str(gpu),
                "engine": engine,
                "model": model,
                "mean_s": sum(timings) / len(timings),
                "min_s": min(timings),
            }
        )
    return rows


def benchmark_yolo(image_path, weights="yolov8n.pt", iterations=10, gpus=None):
    """Benchmark YOLO detection on each selected GPU. Returns list of rows.

    Rows: {gpus, model, mean_ms, min_ms}. warmup run excluded from stats.
    """
    from ai_toolset import detect

    rows = []
    for gpu in gpus or [None]:

        def run_once(gpu=gpu):
            detect.detect_image(
                image_path, weights=weights, gpus=[gpu] if gpu is not None else None
            )

        run_once()  # warmup (weights download + model load)
        timings = _timed(run_once, iterations)
        rows.append(
            {
                "gpus": str(gpu),
                "model": weights,
                "mean_ms": sum(timings) / len(timings) * 1000,
                "min_ms": min(timings) * 1000,
            }
        )
    return rows


def print_table(rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    widths = {
        k: max(
            len(k), *(len(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k])) for r in rows)
        )
        for k in keys
    }
    print("  ".join(k.ljust(widths[k]) for k in keys))
    for r in rows:
        cells = []
        for k in keys:
            v = r[k]
            cells.append((f"{v:.3f}" if isinstance(v, float) else str(v)).ljust(widths[k]))
        print("  ".join(cells))
