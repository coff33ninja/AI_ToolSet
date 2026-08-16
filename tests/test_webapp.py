"""Web dashboard: app factory and live /api/health."""

import json
import socket
import threading
import time
import urllib.request

import pytest
import uvicorn

from ai_toolset.webapp import create_app


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_create_app_routes():
    app = create_app()
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/api/health" in paths
    assert "/api/ocr" in paths


def test_health_live():
    app = create_app()
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}/api/health"
        last = None
        for _ in range(50):
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    payload = json.load(r)
                break
            except Exception as exc:  # noqa: BLE001 - server still booting
                last = exc
                time.sleep(0.2)
        else:
            pytest.fail(f"server did not answer: {last}")
        assert "gpus" in payload
        assert "extras" in payload
    finally:
        server.should_exit = True
        thread.join(timeout=5)
