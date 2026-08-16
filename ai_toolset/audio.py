"""Audio dataset utilities for voice cloning and TTS pipelines.

Designed for the project-local workflow:  uv sync --extra voice

RVC (Retrieval-based Voice Conversion), XTTS v2 and so-vits-svc all want a
folder of clean, resampled speech segments before training. This module
provides the standard preprocessing: probing, resampling to mono WAV,
splitting on silence, and producing the <speaker>_<index>.wav naming
convention the RVC webui expects.
"""

import contextlib
import glob
import os
import threading

import librosa
import soundfile as sf

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus")


def probe_dir(src_dir, exts=AUDIO_EXTS):
    """List audio files with durations.

    Returns a list of dicts: {path, name, duration_sec}.
    """
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(src_dir, f"*{ext}")))
    files.sort()
    out = []
    for path in files:
        try:
            info = sf.info(path)
        except Exception as exc:  # noqa: BLE001 - skip unreadable files
            print(f"  skip {path}: {exc}")
            continue
        out.append({"path": path, "name": os.path.basename(path), "duration_sec": info.duration})
    return out


def resample_dir(src_dir, out_dir, sr=48000, mono=True, exts=AUDIO_EXTS):
    """Resample every audio file to mono WAV (PCM16) at `sr`.

    The RVC webui wants 48 kHz mono for v2 (40 kHz is an alternative).
    Returns the number of files written.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for entry in probe_dir(src_dir, exts=exts):
        y, _ = librosa.load(entry["path"], sr=sr, mono=mono)
        out_path = os.path.join(out_dir, os.path.splitext(entry["name"])[0] + ".wav")
        sf.write(out_path, y, sr, subtype="PCM_16")
        written += 1
    return written


def _split_intervals(y, sr, max_sec=60, min_silence=0.5, top_db=40):
    """Return (start, end) sample-intervals covering speech.

    Splits on silence gaps >= min_silence, then hard-chunks any interval
    longer than max_sec into max_sec pieces (for long monologues). Keeps
    intervals of at least 0.3 s and merges gaps shorter than min_silence.

    top_db is a POSITIVE dB value: frames more than top_db below the loudest
    frame count as silence (librosa.effects.split semantics, not the -35 dB
    RMS convention used by some voice tools).
    """
    max_len = int(max_sec * sr)
    min_len = int(0.3 * sr)
    min_gap = int(min_silence * sr)
    intervals = []
    for start, end in librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512):
        start = int(start)
        end = int(end)
        if end - start < min_len:
            continue
        if intervals and start - intervals[-1][1] < min_gap:
            intervals[-1] = (intervals[-1][0], end)
        else:
            intervals.append((start, end))
    chunks = []
    for start, end in intervals:
        while end - start > max_len:
            chunks.append((start, start + max_len))
            start += max_len
        if end - start >= min_len:
            chunks.append((start, end))
    return chunks


def split_on_silence(
    path, out_dir, sr=48000, max_sec=60, min_silence=0.5, top_db=40, base_name=None
):
    """Split one audio file on silence into clean speech segments (WAV).

    Returns the list of written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    y, _ = librosa.load(path, sr=sr, mono=True)
    stem = base_name or os.path.splitext(os.path.basename(path))[0]
    written = []
    for i, (start, end) in enumerate(_split_intervals(y, sr, max_sec, min_silence, top_db)):
        out_path = os.path.join(out_dir, f"{stem}_{i:03d}.wav")
        sf.write(out_path, y[start:end], sr, subtype="PCM_16")
        written.append(out_path)
    return written


def make_rvc_dataset(
    src_dir, out_dir, speaker_id=0, max_sec=60, sr=48000, min_silence=0.5, top_db=40
):
    """Preprocess a folder of recordings into an RVC-ready dataset.

    Splits every audio file on silence (hard-capping segments at max_sec) and
    writes them as <speaker_id>_<index>.wav, the naming the RVC webui expects
    for preprocessing. Returns the number of segments written.
    """
    os.makedirs(out_dir, exist_ok=True)
    index = 0
    for entry in probe_dir(src_dir):
        y, _ = librosa.load(entry["path"], sr=sr, mono=True)
        for start, end in _split_intervals(y, sr, max_sec, min_silence, top_db):
            out_path = os.path.join(out_dir, f"{speaker_id}_{index:06d}.wav")
            sf.write(out_path, y[start:end], sr, subtype="PCM_16")
            index += 1
    return index


def list_audio_devices():
    """List sounddevice (PortAudio) input/output devices.

    Returns (inputs, outputs, hostapis) tuples; each device is a dict with
    name/max_input_channels/max_output_channels/default_sample_rate.
    """
    import sounddevice as sd

    return sd.query_devices()


def record_mic(out_path, duration, sr=16000, device=None):
    """Record the microphone to a mono WAV file via sounddevice.

    duration>0 records exactly that many seconds (blocking). duration<=0
    records until Enter is pressed in the console. Returns (written_path,
    seconds_recorded). Requires the `audio` extra:  uv sync --extra audio
    """
    import numpy as np
    import sounddevice as sd

    if duration > 0:
        data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32", device=device)
        sd.wait()
        frames = data
        seconds = duration
    else:
        frames = []
        stop = threading.Event()
        counter = {"n": 0}

        def _stop_on_enter():
            with contextlib.suppress(EOFError):
                input("Press Enter to stop recording...\n")
            stop.set()

        def _callback(indata, frames_count, time_info, status):
            frames.append(indata.copy())
            counter["n"] += frames_count

        threading.Thread(target=_stop_on_enter, daemon=True).start()
        with sd.InputStream(
            samplerate=sr, channels=1, dtype="float32", device=device, callback=_callback
        ):
            while not stop.is_set():
                sd.sleep(100)
        data = np.concatenate(frames, axis=0) if frames else np.zeros((0, 1))
        seconds = counter["n"] / sr

    data = data.reshape(-1) if data.ndim > 1 else data
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, data, sr, subtype="PCM_16")
    return out_path, seconds
