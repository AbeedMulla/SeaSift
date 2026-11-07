"""Small helpers for reading video files with OpenCV."""
from pathlib import Path

import cv2

from . import config


def open_capture(path):
    """Open a video, asking for hardware decoding first (if enabled), else normal decoding."""
    if config.HW_DECODE:
        try:
            cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            pass
    return cv2.VideoCapture(str(path))


def video_info(path):
    """Return (fps, frame_count, duration_seconds) for a video file."""
    cap = open_capture(path)
    if not cap.isOpened():
        raise ValueError(f"Can't open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frames, frames / fps


def resize_to_width(frame, width):
    """Shrink a frame to the given width, keeping its shape. Never makes it bigger."""
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def sample_frames(path, every_s, width=None):
    """Yield (time_s, frame) roughly every `every_s` seconds.

    Reads the file once from start to end. Frames we skip are only "grabbed",
    which is faster than decoding them.
    """
    cap = open_capture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps * every_s))
    i = 0
    while True:
        if i % step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            yield i / fps, (resize_to_width(frame, width) if width else frame)
        elif not cap.grab():
            break
        i += 1
    cap.release()


class FrameReader:
    """Jump to any time in a video and read that frame.

    with FrameReader("clip.mp4") as reader:
        frame = reader.at(12.5)
    """

    def __init__(self, path):
        self.cap = open_capture(path)

    def at(self, t_s, width=None):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s) * 1000)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return resize_to_width(frame, width) if width else frame

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def cached_frame(path, t, folder, width=960, quality=90, reader=None):
    """The frame at time t (shrunk to `width`), saved as a JPEG in `folder` the first time.

    Jumping to a time in a 4K video takes about half a second, so each picture is decoded
    once, ever. After that it opens instantly, even after restarting the app.
    reader: an open FrameReader, to reuse when saving many pictures in a row.
    """
    out = Path(folder) / f"t_{t:08.2f}_w{width}.jpg"
    frame = cv2.imread(str(out)) if out.exists() else None
    if frame is None:
        if reader is None:
            with FrameReader(path) as own_reader:
                frame = own_reader.at(t, width=width)
        else:
            frame = reader.at(t, width=width)
        if frame is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return frame
