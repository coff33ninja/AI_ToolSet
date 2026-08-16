"""mss-based screen capture utilities."""

import cv2
import numpy as np
from mss import mss

from ai_toolset.images import pad_to_square

FULL_SCREEN = None


def capture(region=FULL_SCREEN):
    """Capture a screen region as a BGR numpy array.

    region is a dict with top/left/width/height. FULL_SCREEN/None = full
    virtual desktop.
    """
    with mss() as sct:
        if region is None:
            region = sct.monitors[0]
        shot = sct.grab(region)
        img = np.array(shot)
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)


def capture_square(region=FULL_SCREEN, color=(0, 0, 0)):
    """Capture a region and pad it to a square, returning (frame, padding)."""
    frame = capture(region)
    return pad_to_square(frame, color=color)


def select_region(capture_region=FULL_SCREEN, window_name="Select region - drag, Enter=ok, Esc=cancel"):
    """Interactively select a screen region with the mouse.

    Returns a region dict (top/left/width/height) or None on cancel.
    """
    frame = capture(capture_region)
    disp = frame.copy()
    state = {"drawing": False, "start": None, "end": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            state["end"] = (x, y)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    result = None
    while True:
        overlay = disp.copy()
        if state["start"] and state["end"]:
            cv2.rectangle(overlay, state["start"], state["end"], (0, 255, 0), 1)
        cv2.imshow(window_name, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 32) and state["start"] and state["end"]:
            x1, y1 = min(state["start"][0], state["end"][0]), min(state["start"][1], state["end"][1])
            x2, y2 = max(state["start"][0], state["end"][0]), max(state["start"][1], state["end"][1])
            result = {"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1}
            break
        if key == 27:
            break
    cv2.destroyWindow(window_name)
    return result


def stream_frames(region=FULL_SCREEN):
    """Yield captured frames indefinitely. FULL_SCREEN/None = full desktop."""
    while True:
        yield capture(region)
