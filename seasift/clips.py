"""Enhanced video clips of an event, with orange boxes where SeaSift saw movement.

Used by the Review page ("Play this event as video") and tools/make_demo_clip.py.
Classical computer vision only: the same motion triage and enhancement as the rest of the app.
The triage here watches every frame it writes (about CLIP_FPS a second), so the boxes move
smoothly. It's the same method that found the event, just looking more often.
"""
import json
from pathlib import Path

import cv2

from . import config
from .enhance import enhance
from .triage import MotionTriage
from .video import open_capture, resize_to_width
from .visibility import frame_visibility, label_for

ORANGE = (0, 140, 255)  # BGR. Easy to see on green or blue water, and on enhanced (greyer) frames.

# Video formats to try, best first. Browsers play H.264 MP4 and WebM (VP8).
FOR_BROWSER = [("avc1", ".mp4"), ("VP80", ".webm")]
# For slides and video players: H.264 first. mp4v plays in QuickTime and VLC, but not in browsers.
FOR_SLIDES = [("avc1", ".mp4"), ("mp4v", ".mp4"), ("VP80", ".webm")]
FORMAT_NAMES = {"avc1": "H.264 MP4", "VP80": "WebM", "mp4v": "MPEG-4 (mp4v)"}


def read_boxes(text):
    """Movement boxes as saved in the events table -> list of (x, y, w, h). Empty if none."""
    if not isinstance(text, str) or not text:
        return []
    return [tuple(box) for box in json.loads(text)]


def draw_movement(frame, boxes, color=ORANGE):
    """Draw movement boxes (x, y, w, h as shares of the frame) on a copy of the frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    thickness = max(2, round(w / 250))
    for x, y, bw, bh in boxes:
        cv2.rectangle(out, (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h)),
                      color, thickness)
    return out


def put_label(img, text, x, y):
    """White text on a dark band, so it's readable on any footage."""
    scale = max(0.45, img.shape[0] / 800)
    thick = max(1, round(scale * 2))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + base + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


def side_by_side(original, better, score, blocked):
    """Original on the left, enhanced on the right, with titles and the footage-quality score."""
    both = cv2.hconcat([original, better])
    margin = max(10, both.shape[0] // 30)
    put_label(both, "Original", margin, margin * 2)
    put_label(both, "Enhanced by SeaSift (orange = movement)", original.shape[1] + margin, margin * 2)
    put_label(both, f"Footage quality {score:.0f}/100: {label_for(score, blocked)}",
              margin, both.shape[0] - margin)
    return both


def open_writer(out_base, fps, size, formats):
    """Try each video format until one works on this computer. Returns (writer, path, fourcc)."""
    for fourcc, ext in formats:
        path = Path(out_base).with_suffix(ext)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path, fourcc
        writer.release()
        path.unlink(missing_ok=True)
    return None, None, None


def make_clip(video_path, start_s, end_s, out_base, compare=False, boxes=True,
              width=640, fps_out=config.CLIP_FPS, formats=FOR_BROWSER, speed=1.0, enhanced=True):
    """Write an enhanced clip of start_s..end_s, with orange boxes where something moved.

    compare=True puts the original on the left and the enhanced version on the right.
    enhanced=False keeps the original colours (a small clip, instead of the whole video file).
    speed=0.25 plays it at quarter speed (for counting, as the protocol suggests).
    out_base: file path without the extension (the format decides .mp4 or .webm).
    Returns (path, format name), or (None, None) if this computer can't write any of the formats.
    """
    cap = open_capture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / fps_out))  # keep about fps_out frames per second
    first = int(max(0.0, start_s - config.CLIP_LEAD_S) * fps)  # start watching a bit earlier
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    triage = MotionTriage()
    writer = path = fourcc = None
    i = first
    while True:
        if (i - first) % step:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        t = i / fps
        i += 1
        if not ok or t > end_s:
            break
        triage.step(resize_to_width(frame, config.PROCESS_WIDTH))
        if t < start_s:
            continue  # still learning what the empty scene looks like
        view = resize_to_width(frame, width)
        view = view[: view.shape[0] // 2 * 2, : view.shape[1] // 2 * 2]  # video formats want even sizes
        better = enhance(view) if enhanced else view.copy()
        better = draw_movement(better, triage.boxes) if boxes else better
        out = side_by_side(view, better, *frame_visibility(view)) if compare else better
        if writer is None:
            Path(out_base).parent.mkdir(parents=True, exist_ok=True)
            writer, path, fourcc = open_writer(out_base, fps / step * speed, (out.shape[1], out.shape[0]),
                                               formats)
            if writer is None:
                break
        writer.write(out)
    cap.release()
    if writer is None:
        return None, None
    writer.release()
    return path, FORMAT_NAMES.get(fourcc, fourcc)
