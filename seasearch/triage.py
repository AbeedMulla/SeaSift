"""Motion triage: find the moments where something moves, skip everything else.

Classical computer vision, no trained model: OpenCV's MOG2 background subtractor
(a per-pixel statistics model that adapts as the video plays) marks pixels that changed.
Then we throw away the things that aren't fish:
  - tiny specks (sediment)                      -> MIN_BLOB_FRAC
  - the whole frame changing at once (bump, sediment cloud)         -> MAX_FG_FRAC
  - the water getting murkier/clearer or the light changing         -> GLOBAL_CHANGE
  - pixels that move all the time (swaying eelgrass, kelp)          -> the "always moving" map
  - one-frame flickers                          -> PERSIST_WINDOW / PERSIST_MIN

After every step(), .boxes says where the kept movement was: a list of (x, y, w, h)
as shares of the frame (0 to 1), so they can be drawn on a picture of any size.
"""
from collections import deque

import cv2
import numpy as np

from . import config


class MotionTriage:
    """Feed it frames one by one with step(); it tells you if real motion is happening."""

    def __init__(self):
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=config.BG_HISTORY, varThreshold=config.BG_THRESHOLD, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.always_moving = None              # running average of "this pixel moved"
        self.recent = deque(maxlen=config.PERSIST_WINDOW)
        self.looks = deque(maxlen=config.TRIAGE_FPS + 1)  # (brightness, contrast) over ~1 second
        self.boxes = []                        # where the kept movement was in the last frame

    def scene_changing(self, gray):
        """True if the whole picture is getting brighter/darker or murkier/clearer right now."""
        self.looks.append((float(gray.mean()), float(gray.std())))
        if len(self.looks) < self.looks.maxlen:
            return False
        (b0, c0), (b1, c1) = self.looks[0], self.looks[-1]
        change = max(abs(b1 - b0) / max(b0, 1.0), abs(c1 - c0) / max(c0, 1.0))
        return change > config.GLOBAL_CHANGE

    def step(self, frame):
        """Returns (active, motion_share) for this frame."""
        self.boxes = []
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if self.scene_changing(gray):
            # Re-learn the background quickly, and don't trust motion in this frame
            self.background.apply(gray, learningRate=0.3)
            self.recent.append(False)
            return False, 0.0
        mask = self.background.apply(gray)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)  # remove specks
        moving = mask > 0

        # Learn which pixels move all the time (grass that sways in the same spot)
        if self.always_moving is None:
            self.always_moving = np.zeros(moving.shape, np.float32)
        a = config.CHRONIC_ALPHA
        self.always_moving = (1 - a) * self.always_moving + a * moving

        if moving.mean() > config.MAX_FG_FRAC:
            # Most of the picture changed at once: not a fish
            self.recent.append(False)
            return False, 0.0

        grass = self.always_moving > config.CHRONIC_LEVEL
        contours, _ = cv2.findContours(moving.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        min_area = config.MIN_BLOB_FRAC * moving.size
        height, width = moving.shape
        motion_area, boxes = 0.0, []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            blob = moving[y:y + h, x:x + w]
            in_grass = (blob & grass[y:y + h, x:x + w]).sum() / max(1, blob.sum())
            if in_grass > config.MAX_CHRONIC_SHARE:
                continue
            motion_area += area
            boxes.append((x / width, y / height, w / width, h / height))

        self.boxes = merge_boxes(boxes)
        self.recent.append(motion_area > 0)
        active = sum(self.recent) >= config.PERSIST_MIN
        return active, motion_area / moving.size


def merge_boxes(boxes, gap=0.02):
    """Join boxes that overlap or almost touch, so one fish gets one box instead of several.

    boxes: list of (x, y, w, h) as shares of the frame. gap: how close counts as touching.
    """
    boxes = [list(b) for b in boxes]
    joined = True
    while joined:
        joined = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if (a[0] - gap < b[0] + b[2] and b[0] - gap < a[0] + a[2]
                        and a[1] - gap < b[1] + b[3] and b[1] - gap < a[1] + a[3]):
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    del boxes[j]
                    joined = True
                    break
            if joined:
                break
    return [tuple(round(v, 4) for v in b) for b in boxes]


def build_events(samples):
    """Turn per-frame results into events.

    samples: list of (time_s, active, motion_share)
    returns: list of {"start_s", "end_s", "peak_s", "motion"}
    """
    events, current = [], None
    for t, active, motion in samples:
        if t < config.WARMUP_S:
            continue
        if active:
            if current is None:
                current = {"start_s": t, "end_s": t, "peak_s": t, "motion": motion}
            current["end_s"] = t
            if motion > current["motion"]:
                current["peak_s"], current["motion"] = t, motion
        elif current is not None and t - current["end_s"] > config.MERGE_GAP_S:
            events.append(current)
            current = None
    if current is not None:
        events.append(current)

    kept = []
    for ev in events:
        if ev["end_s"] - ev["start_s"] < config.MIN_EVENT_S:
            continue
        ev["start_s"] = max(0.0, ev["start_s"] - config.PAD_S)
        ev["end_s"] = ev["end_s"] + config.PAD_S
        ev["motion"] = round(ev["motion"], 4)
        kept.append(ev)
    return kept
