"""Video frame extraction and video stitching utilities."""

import glob
import os
import random

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
