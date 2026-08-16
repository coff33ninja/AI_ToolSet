"""MediaPipe landmark detection (face/hands/pose/holistic) + selfie segmentation.

Wraps mediapipe.solutions so callers get a uniform interface: process a BGR
frame, annotate it with the detected landmarks, and stream annotated webcam
frames. Everything is imported lazily so the module is importable even when
the mediapipe extra is not installed.
"""

import cv2

SOLUTIONS = (
    "pose",
    "hands",
    "face",
    "holistic",
    "selfie",
)

_LANDMARK_COLORS = {
    "pose": (0, 255, 0),
    "hands": (255, 200, 0),
    "face": (200, 0, 255),
}


def available():
    """Return True if the mediapipe extra is importable."""
    try:
        import mediapipe  # noqa: F401

        return True
    except ImportError:
        return False


def _solutions_module():
    import mediapipe as mp

    return mp.solutions


def _process_spec(solution, static, min_conf):
    """Return (factory_kwargs, results_attr, connections, is_mask) for a solution."""
    mp = _solutions_module()
    if solution == "pose":
        return (
            {"static_image_mode": static, "min_detection_confidence": min_conf},
            "pose_landmarks",
            mp.pose.POSE_CONNECTIONS,
            False,
        )
    if solution == "hands":
        return (
            {"static_image_mode": static, "min_detection_confidence": min_conf},
            "multi_hand_landmarks",
            mp.hands.HAND_CONNECTIONS,
            False,
        )
    if solution == "face":
        return (
            {"static_image_mode": static, "min_detection_confidence": min_conf,
             "min_tracking_confidence": min_conf},
            "face_landmarks",
            mp.face_mesh.FACEMESH_TESSELATION,
            False,
        )
    if solution == "holistic":
        return (
            {"static_image_mode": static, "min_detection_confidence": min_conf},
            "holistic",
            None,
            False,
        )
    if solution == "selfie":
        return (
            {"model_selection": 0},
            "segmentation_mask",
            None,
            True,
        )
    raise ValueError(f"Unknown solution '{solution}'; choose from {SOLUTIONS}")


def _make_solution(solution, static=False, min_conf=0.5):
    mp = _solutions_module()
    kwargs, _, _, _ = _process_spec(solution, static, min_conf)
    if solution == "pose":
        return mp.pose.Pose(**kwargs)
    if solution == "hands":
        return mp.hands.Hands(**kwargs)
    if solution == "face":
        return mp.face_mesh.FaceMesh(**kwargs)
    if solution == "holistic":
        return mp.holistic.Holistic(**kwargs)
    if solution == "selfie":
        return mp.selfie_segmentation.SelfieSegmentation(**kwargs)


def process_frame(frame, solution="pose", static=False, min_conf=0.5):
    """Run one solution on a BGR frame. Returns the mediapipe results object."""
    solution = _make_solution(solution, static=static, min_conf=min_conf)
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return solution.process(rgb)
    finally:
        solution.close()


def _draw_landmarks(frame, landmarks, connections, color):
    if landmarks is None:
        return
    mp = _solutions_module()
    drawing = mp.drawing_utils
    spec = drawing.DrawingSpec(color=color, thickness=1, circle_radius=1)
    if isinstance(landmarks, list):
        for lm in landmarks:
            drawing.draw_landmarks(frame, lm, connections, spec, spec)
    else:
        drawing.draw_landmarks(frame, landmarks, connections, spec, spec)


def _draw_holistic(frame, results):
    mp = _solutions_module()
    drawing = mp.drawing_utils
    style = drawing.DrawingSpec
    pose_spec = style(color=_LANDMARK_COLORS["pose"], thickness=2, circle_radius=2)
    hand_spec = style(color=_LANDMARK_COLORS["hands"], thickness=2, circle_radius=2)
    face_spec = style(color=_LANDMARK_COLORS["face"], thickness=1, circle_radius=1)
    if results.pose_landmarks:
        drawing.draw_landmarks(frame, results.pose_landmarks,
                               mp.pose.POSE_CONNECTIONS,
                               pose_spec, pose_spec)
    if results.face_landmarks:
        drawing.draw_landmarks(frame, results.face_landmarks,
                               mp.face_mesh.FACEMESH_CONTOURS,
                               face_spec, face_spec)
    for lm in results.left_hand_landmarks or []:
        drawing.draw_landmarks(frame, lm, mp.hands.HAND_CONNECTIONS,
                               hand_spec, hand_spec)
    for lm in results.right_hand_landmarks or []:
        drawing.draw_landmarks(frame, lm, mp.hands.HAND_CONNECTIONS,
                               hand_spec, hand_spec)


def annotate(frame, solution, results, mask_threshold=0.5, mask_color=(80, 160, 255)):
    """Draw a solution's results onto a BGR frame (in place). Returns the frame."""
    if solution == "selfie":
        mask = getattr(results, "segmentation_mask", None)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            binary = (mask > mask_threshold).astype("uint8")
            overlay = frame.copy()
            overlay[:] = mask_color
            frame[:] = cv2.addWeighted(
                frame, 0.4, cv2.bitwise_and(overlay, overlay, mask=binary), 0.6, 0
            )
        return frame
    if solution == "holistic":
        _draw_holistic(frame, results)
        return frame
    attr, connections, _ = _process_spec(solution, False, 0.5)[1:]
    landmarks = getattr(results, attr, None)
    if landmarks is not None:
        _draw_landmarks(frame, landmarks, connections,
                        _LANDMARK_COLORS.get(solution, (0, 255, 0)))
    return frame


def overlay(frame, solution="pose", static=False, min_conf=0.5,
            mask_threshold=0.5, mirror=True):
    """Run a solution on a frame and return the annotated BGR frame."""
    if mirror:
        frame = cv2.flip(frame, 1)
    results = process_frame(frame, solution=solution, static=static,
                            min_conf=min_conf)
    return annotate(frame, solution, results, mask_threshold=mask_threshold)


def webcam_stream(camera=0, solution="pose", mirror=True, min_conf=0.5,
                  mask_threshold=0.5):
    """Yield annotated BGR frames from a webcam. Consume with cv2.imshow."""
    if not available():
        raise RuntimeError("mediapipe not installed: uv sync --extra mediapipe")
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise OSError(f"Could not open camera {camera}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield overlay(frame, solution=solution, mirror=mirror,
                          min_conf=min_conf, mask_threshold=mask_threshold)
    finally:
        cap.release()
