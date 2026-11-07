"""How usable is the footage? A score from 0 (can't see anything) to 100 (crystal clear).

This is a relative footage-quality score, NOT a turbidity measurement (no NTU).
Calibrate the thresholds in config.py on Coastal Action's own clips.

No trained model here, just two simple measurements per frame:
  contrast  = how spread out the brightness values are (murky water looks flat and grey)
  sharpness = how strong the edges are (sediment and turbidity blur everything)
"""
import math
import statistics

import cv2

from . import config


def _scale(value, low, high):
    """Map value onto 0..1 between low and high."""
    return min(1.0, max(0.0, (value - low) / (high - low)))


def frame_visibility(frame):
    """Return (score 0..100, blocked) for one frame. blocked = lens covered or far too dark/bright."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    # Measure on a small copy: shrinking averages away sensor noise and sediment specks,
    # so they don't count as "sharp edges", while real edges (rocks, fish) survive.
    small = cv2.resize(gray, (160, max(1, int(gray.shape[0] * 160 / gray.shape[1]))),
                       interpolation=cv2.INTER_AREA)
    contrast = float(small.std())
    sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())

    c = _scale(contrast, *config.CONTRAST_RANGE)
    low, high = config.SHARPNESS_RANGE
    s = _scale(math.log(sharpness + 1), math.log(low + 1), math.log(high + 1))
    score = round(100 * (0.5 * c + 0.5 * s), 1)
    blocked = brightness < config.DARK_LEVEL or brightness > config.BRIGHT_LEVEL
    return score, blocked


def label_for(score, blocked=False):
    if blocked:
        return "blocked"
    if score >= config.CLEAR_SCORE:
        return "clear"
    if score >= config.MURKY_SCORE:
        return "murky"
    return "unusable"


def summarize_windows(samples, duration_s, window_s=config.VIS_WINDOW_S):
    """Group per-frame scores into windows.

    samples: list of (time_s, score, blocked)
    returns: list of {"start_s", "end_s", "score", "label"}
    """
    buckets = {}
    for t, score, blocked in samples:
        buckets.setdefault(int(t // window_s), []).append((score, blocked))

    segments = []
    for k in sorted(buckets):
        values = buckets[k]
        score = statistics.median(s for s, _ in values)
        blocked = sum(b for _, b in values) > len(values) / 2
        start = k * window_s
        end = min(start + window_s, duration_s) if duration_s else start + window_s
        segments.append({"start_s": start, "end_s": max(end, start),
                         "score": round(score, 1), "label": label_for(score, blocked)})
    return segments


def minutes_by_label(segments):
    """How many minutes were clear / murky / unusable / blocked."""
    totals = {"clear": 0.0, "murky": 0.0, "unusable": 0.0, "blocked": 0.0}
    for seg in segments:
        totals[seg["label"]] += (seg["end_s"] - seg["start_s"]) / 60
    return {k: round(v, 1) for k, v in totals.items()}
