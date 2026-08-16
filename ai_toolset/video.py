"""Video frame extraction, video stitching, and screen/webcam recording."""

import glob
import os
import random
import time

import cv2


def extract_frames(video_path, out_dir, mode="random", count=0, min_skip=100,
                   max_skip=2500, ext="jpg", start_index=0):
    """Extract frames from a video.

    mode "random": skip a random number of frames (min_skip..max_skip) between
    captures. mode "interval": capture every min_skip frames.
    count 0 = extract until the video ends.
    Returns the number of frames written.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    index = start_index
    saved = 0
    while True:
        if mode == "random":
            skip = random.randint(min_skip, max_skip)
            for _ in range(skip):
                if not cap.grab():
                    cap.release()
                    return saved
        else:
            for _ in range(max(0, min_skip - 1)):
                if not cap.grab():
                    cap.release()
                    return saved
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f"{index:05d}.{ext}"), frame)
        index += 1
        saved += 1
        if count and saved >= count:
            break
    cap.release()
    return saved


def _numeric_key(path):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(name)
    except ValueError:
        return name


def frames_to_video(frames_dir, out_path, fps=30, ext="jpg", codec="mp4v", size=None):
    """Stitch all frames in a directory into a video.

    Frames sort by numeric filename when possible, else lexically.
    Returns the number of frames written.
    """
    paths = glob.glob(os.path.join(frames_dir, f"*.{ext}"))
    if not paths:
        raise FileNotFoundError(f"No .{ext} files in {frames_dir}")
    paths.sort(key=_numeric_key)
    first = cv2.imread(paths[0])
    if first is None:
        raise OSError(f"Could not read {paths[0]}")
    h, w = first.shape[:2]
    if size:
        w, h = size
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    written = 0
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        if size:
            img = cv2.resize(img, size)
        writer.write(img)
        written += 1
    writer.release()
    return written


def _video_writer(out_path, fps, size, codec="mp4v"):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    return cv2.VideoWriter(out_path, fourcc, fps, size)


def record_screen(region, out_path, duration=0, fps=20, codec="mp4v"):
    """Record a screen region to a video file.

    region is a dict (top/left/width/height) from screen.select_region, or
    screen.FULL_SCREEN. duration=0 records until ESC is pressed (video window
    must be focused). Returns the number of frames written.
    """
    from ai_toolset.screen import capture

    first = capture(region)
    h, w = first.shape[:2]
    writer = _video_writer(out_path, fps, (w, h), codec)
    start = time.time()
    frames = 0
    print("Recording screen... press ESC in the preview window to stop.")
    try:
        while True:
            frame = capture(region)
            writer.write(frame)
            frames += 1
            cv2.imshow("recording", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            if duration and time.time() - start >= duration:
                break
    finally:
        writer.release()
        cv2.destroyWindow("recording")
    return frames


def webcam_capture(camera=0, out_path=None, duration=0, fps=20, codec="mp4v",
                   save_dir=None):
    """Show the webcam with record/snapshot controls.

    Keys in the preview window: r = toggle recording, s = save a snapshot JPG,
    q/ESC = quit. out_path (recording destination) defaults to webcam_<ts>.mp4;
    save_dir defaults to ./snapshots. Returns (frames_recorded, snapshots_saved).
    """
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise OSError(f"Could not open camera {camera}")
    ret, first = cap.read()
    if not ret:
        cap.release()
        raise OSError(f"Camera {camera} returned no frames")
    h, w = first.shape[:2]
    out_path = out_path or f"webcam_{int(time.time())}.mp4"
    save_dir = save_dir or os.path.join("snapshots")
    os.makedirs(save_dir, exist_ok=True)
    writer = None
    recorded = 0
    snapshots = 0
    start = time.time()
    print("r = record toggle, s = snapshot, q/ESC = quit")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if writer is not None:
                writer.write(frame)
                recorded += 1
            cv2.putText(frame, "REC" if writer else "idle",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255) if writer else (0, 255, 0), 2)
            cv2.imshow("webcam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                if writer is not None:
                    writer.release()
                    writer = None
                    print(f"Recording saved: {out_path}")
                else:
                    writer = _video_writer(out_path, fps, (w, h), codec)
                    print(f"Recording started: {out_path}")
            if key == ord("s"):
                snap_path = os.path.join(save_dir, f"snap_{int(time.time())}.jpg")
                cv2.imwrite(snap_path, frame)
                snapshots += 1
                print(f"Snapshot: {snap_path}")
            if duration and time.time() - start >= duration:
                break
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyWindow("webcam")
    return recorded, snapshots
