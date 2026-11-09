"""Coastal Action's video analysis protocol (MeanCount), done step by step. No AI.

From their DRAFT FFH Video Analysis Protocol:
  - skip the first and last minute (sediment settling), and a few more if it hasn't settled
  - analyse a 15-minute window: one sub-sample frame every 60 seconds (15 frames)
  - count fish and invertebrates in each frame, watching 5 s before and after for context
  - MeanCount = the mean number of fish per sub-sample frame
  - quality check: at least 10% of frames counted again, blind, by a second person;
    the two counts should differ by less than 6%

seaSEArch does the bookkeeping: it picks the frames, shows them enhanced with a grid, lets you
click each fish, adds everything up and writes their spreadsheet. People still do the counting.
"""
import math
import random
from collections import Counter

import cv2
import pandas as pd

from . import config

VISIBILITY = ["<0.5 m", "0.5-1.0 m", "1.0-1.5 m", ">1.5 m"]  # their horizontal visibility categories
KINDS = {"fish": "Fish", "unknown": "Fish, can't tell the species", "invert": "Invertebrate (crab, shrimp...)"}
DOT_COLORS = {"fish": (0, 220, 255), "unknown": (255, 255, 0), "invert": (255, 0, 255)}  # BGR


def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def max_extra_skip(duration_s):
    """How many minutes after the first one the protocol lets you skip while the footage settles."""
    if duration_s > config.VERY_LONG_VIDEO_S:
        return config.MAX_EXTRA_SKIP_VERY_LONG_S
    if duration_s > config.LONG_VIDEO_S:
        return config.MAX_EXTRA_SKIP_LONG_S
    return config.MAX_EXTRA_SKIP_S


def bad_frame(t, segments, events):
    """True if a count frame at time t lands in unusable footage or a big disturbance."""
    murky = any(s["label"] in ("unusable", "blocked") and s["start_s"] <= t < s["end_s"] for s in segments)
    busy = any(e["motion"] >= config.SETTLING_MOTION and overlaps(t - config.CONTEXT_S, t + config.CONTEXT_S,
                                                                    e["start_s"], e["end_s"]) for e in events)
    return murky or busy


def suggest_window(duration_s, segments, events, every_s=config.SUBSAMPLE_EVERY_S):
    """Suggest where the analysis window starts, and how many frames fit.

    Skip the first minute, then try later starts (whole minutes, up to the protocol's limit)
    and pick the earliest one where the fewest count frames land in unusable footage
    (sediment still settling, camera being set up).
    segments, events: lists of dicts from the database (label / motion with start_s, end_s).
    Returns (start_s, number of frames, how many of those frames land in unusable footage).
    """
    last_start = max(0.0, duration_s - config.SKIP_END_S - every_s)  # short clips: never start past the end
    first = min(config.SKIP_START_S, last_start)
    starts = [first + m * 60 for m in range(int(max_extra_skip(duration_s) // 60) + 1)
              if first + m * 60 <= last_start] or [first]

    def score(start):
        room = duration_s - config.SKIP_END_S - start
        frames = max(1, min(config.SUBSAMPLE_FRAMES, int(room // every_s) + 1))
        bad = sum(bad_frame(t, segments, events) for t in subsample_times(start, frames, every_s, duration_s))
        return bad, -frames, start  # fewest bad frames, then the most frames, then the earliest

    best = min(starts, key=score)
    bad, minus_frames, _ = score(best)
    return float(best), -minus_frames, bad


def subsample_times(start_s, frames, every_s, duration_s):
    """Equally spaced frame times, inside the video."""
    return [start_s + i * every_s for i in range(frames) if start_s + i * every_s < duration_s]


def nearest_dot(dots, x, y, aspect, radius=config.DOT_RADIUS):
    """The dot a click landed on, if any (dots: list of dicts with x, y as shares of the frame)."""
    best, best_d = None, radius
    for dot in dots:
        d = math.hypot((dot["x"] - x) * aspect, dot["y"] - y)
        if d <= best_d:
            best, best_d = dot, d
    return best


def grid_columns(aspect, rows=config.GRID_ROWS):
    """Number of grid columns when the cells are square and a `rows`-th of the frame high."""
    return math.ceil(rows * aspect - 1e-6)


def grid_counts(dots, aspect=16 / 9, rows=config.GRID_ROWS):
    """How many dots in each grid cell, like the numbers Coastal Action writes on each frame.

    aspect: frame width / height. Cells are square, like the grid on Coastal Action's frames.
    """
    cols = grid_columns(aspect, rows)
    table = [[0] * cols for _ in range(rows)]
    for dot in dots:
        table[min(int(dot["y"] * rows), rows - 1)][min(int(dot["x"] * rows * aspect), cols - 1)] += 1
    return table


def frame_counts(dots):
    """Totals for one frame: fish (named + unknown), unknown fish, invertebrates, and per species."""
    named = Counter(d["species"] for d in dots if d["species"] and d["kind"] in ("fish", "invert"))
    return {
        "fish": sum(d["kind"] in ("fish", "unknown") for d in dots),
        "unknown": sum(d["kind"] == "unknown" for d in dots),
        "invertebrates": sum(d["kind"] == "invert" for d in dots),
        "species": dict(named),
    }


def mean_count(fish_counts):
    """MeanCount: the mean number of fish per counted sub-sample frame."""
    return sum(fish_counts) / len(fish_counts) if fish_counts else 0.0


def percent_difference(a, b):
    """Percent difference between two counts of the same frame: |a - b| / mean(a, b) x 100."""
    if a == b:
        return 0.0
    return abs(a - b) / ((a + b) / 2) * 100


def pick_qc_frames(numbers, share=config.QC_SHARE, minimum=config.QC_MIN_FRAMES, seed=None):
    """Randomly pick frames for the blind second count: at least `share` of them, and `minimum`."""
    k = min(len(numbers), max(minimum, math.ceil(share * len(numbers))))
    return sorted(random.Random(seed).sample(list(numbers), k))


def draw(frame, dots, grid=True, rows=config.GRID_ROWS):
    """The frame with the grid, a dot on every counted animal, and the count in each grid cell."""
    out = frame.copy()
    h, w = out.shape[:2]
    cell = h / rows  # square cells
    if grid:
        for c in range(1, grid_columns(w / h, rows)):
            cv2.line(out, (round(c * cell), 0), (round(c * cell), h), (25, 25, 25), max(1, w // 800))
        for r in range(1, rows):
            cv2.line(out, (0, round(r * cell)), (w, round(r * cell)), (25, 25, 25), max(1, w // 800))
    radius = max(4, w // 260)
    for dot in dots:
        center = (int(dot["x"] * w), int(dot["y"] * h))
        cv2.circle(out, center, radius + 1, (0, 0, 0), -1)
        cv2.circle(out, center, radius, DOT_COLORS.get(dot["kind"], (255, 255, 255)), -1)
    if grid:  # the number in each cell counts fish, like on Coastal Action's frames
        scale = max(0.6, w / 1400)
        fish = [d for d in dots if d["kind"] in ("fish", "unknown")]
        for r, row in enumerate(grid_counts(fish, w / h, rows)):
            for c, n in enumerate(row):
                if n:
                    x, y = round(c * cell) + 8, round(r * cell) + int(34 * scale)
                    cv2.putText(out, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
                    cv2.putText(out, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2,
                                cv2.LINE_AA)
    return out


def text(value):
    """Empty string for missing values (None or NaN from the database)."""
    return "" if value is None or pd.isna(value) else str(value)


def fmt_time(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def spreadsheet(video_name, subsamples, dots, common=None):
    """One row per sub-sample frame, with Coastal Action's spreadsheet columns.

    subsamples: DataFrame (n, time_s, visibility, notes, counter, done)
    dots: DataFrame of the first-count dots (subsample_id, kind, species)
    common: optional function, scientific name -> column heading (e.g. the common name)
    Frames not counted yet get empty counts, not zeros.
    """
    common = common or (lambda name: name)
    per_frame = {}
    for s in subsamples.itertuples():
        frame_dots = dots[dots.subsample_id == s.id].to_dict("records") if not dots.empty else []
        per_frame[s.id] = frame_counts(frame_dots)
    names = sorted({name for c in per_frame.values() for name in c["species"]})
    counts = ["Total Fish Count", "Invertebrate Count"] + [common(n) for n in names] + \
             ["Unknown/Unable to ID due to visibility", "Total species observed"]
    rows = []
    for s in subsamples.itertuples():
        c = per_frame[s.id]
        row = {"Video ID": video_name, "Sub-sample #": int(s.n), "Video time": fmt_time(s.time_s),
               "Counted": "yes" if s.done else "no", "Visibility": text(s.visibility),
               "Total Fish Count": c["fish"], "Invertebrate Count": c["invertebrates"]}
        row.update({common(name): c["species"].get(name, 0) for name in names})
        row.update({"Unknown/Unable to ID due to visibility": c["unknown"],
                    "Total species observed": len(c["species"]),
                    "Notes": text(s.notes), "Counted by": text(s.counter)})
        if not s.done:
            row.update({col: None for col in counts})
        rows.append(row)
    sheet = pd.DataFrame(rows)
    for col in counts:
        if col in sheet:
            sheet[col] = sheet[col].astype("Int64")
    return sheet
