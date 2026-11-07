"""Make a fake murky underwater video with fish at known times, so you can test SeaSift
before you have real footage. It also writes the "answer key" (ground truth).

Usage:  python tools/make_test_video.py
Makes:  data/videos/test_murky.mp4 and data/videos/test_murky_truth.csv

What's in the 90-second clip:
  - swaying eelgrass the whole time (should NOT become events)
  - drifting sediment specks (should NOT become events)
  - 0-30 s clear, 30-60 s murky, 60-70 s very murky, 70-80 s lens blocked, 80-90 s clear
  - a sediment cloud flash at 50 s (should NOT become an event)
  - 5 fish passages, one of them with 2 fish together (so true MaxN = 2)
"""
import csv
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "videos" / "test_murky.mp4"
TRUTH = ROOT / "data" / "videos" / "test_murky_truth.csv"
W, H, FPS, SECONDS = 640, 360, 15, 90

# start_s, end_s, number of fish swimming together, height in frame, direction (1 = left to right)
FISH = [(8, 12, 1, 150, 1), (20, 25, 2, 185, -1), (40, 44, 1, 170, 1),
        (63, 67, 1, 190, -1), (83, 87, 1, 160, 1)]
# (time_s, murk 0..1). Murk changes smoothly between these points, like real water.
MURK_POINTS = [(0, 0.15), (27, 0.15), (35, 0.6), (56, 0.6), (61, 0.9), (70, 0.9),
               (70.01, 0.15), (90, 0.15)]
BLOCKED = (70, 80)              # lens covered / camera tipped into the mud
SEDIMENT_CLOUD = (50.0, 50.6)   # a quick sediment flash
WATER = np.array([95, 120, 70], np.float32)  # murky green-brown (BGR)


def background(rng):
    """A textured seabed: sand grain, ripples and rocks with clear edges."""
    y = np.linspace(0, 1, H)[:, None, None]
    img = (1 - y) * np.array([110, 115, 70]) + y * np.array([70, 95, 105])
    img = np.repeat(img, W, axis=1).astype(np.float32)
    grain = rng.normal(0, 18, (H // 4, W // 4)).astype(np.float32)
    img += cv2.resize(grain, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    for row in range(H // 2, H, 14):  # sand ripples
        cv2.line(img, (0, row), (W, row + int(rng.integers(-6, 6))), (55, 75, 80), 2)
    for _ in range(30):  # rocks
        cx, cy = int(rng.integers(0, W)), int(rng.integers(H // 3, H))
        r = int(rng.integers(6, 26))
        shade = float(rng.integers(30, 90))
        cv2.circle(img, (cx, cy), r, (shade, shade + 10, shade + 5), -1)
        cv2.circle(img, (cx, cy), r, (20, 25, 25), 2)
    return img


def murk_at(t):
    """Smoothly interpolate the murk level at time t."""
    times, levels = zip(*MURK_POINTS)
    return float(np.interp(t, times, levels))


def draw_fish(img, x, y, direction, scale=1.0):
    body = (int(28 * scale), int(11 * scale))
    cv2.ellipse(img, (int(x), int(y)), body, 0, 0, 360, (165, 175, 185), -1)
    tail_x = x - direction * body[0]
    pts = np.array([[tail_x, y], [tail_x - direction * 16 * scale, y - 10 * scale],
                    [tail_x - direction * 16 * scale, y + 10 * scale]], np.int32)
    cv2.fillPoly(img, [pts], (150, 160, 170))
    cv2.circle(img, (int(x + direction * body[0] * 0.6), int(y - 3)), 2, (30, 30, 30), -1)


def main():
    rng = np.random.default_rng(7)
    base = background(rng)
    blades = [(x, int(rng.integers(60, 140)), float(rng.uniform(0, 6.28)))
              for x in np.linspace(20, W - 20, 16)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    for i in range(FPS * SECONDS):
        t = i / FPS
        img = base.copy()

        # Eelgrass swaying in place
        for x, height, phase in blades:
            sway = 14 * math.sin(2 * math.pi * 0.4 * t + phase)
            pts = np.array([[x + sway * (k / 10) ** 2, H - height * k / 10] for k in range(11)], np.int32)
            cv2.polylines(img, [pts], False, (40, 95, 45), 5)

        # Fish swimming across
        for start, end, n, y, direction in FISH:
            if start <= t < end:
                p = (t - start) / (end - start)
                x = -60 + p * (W + 120) if direction > 0 else W + 60 - p * (W + 120)
                for k in range(n):
                    draw_fish(img, x - direction * k * 70, y + k * 40, direction)

        # Sediment specks drifting
        for _ in range(70):
            cv2.circle(img, (int(rng.integers(0, W)), int(rng.integers(0, H))), 1, (150, 170, 175), -1)

        # Turbidity: mix with murky water colour, blur, add a little sensor noise
        k = murk_at(t)
        img = (1 - k) * img + k * WATER
        blur = 1 + 2 * int(k * 7)
        if blur > 1:
            img = cv2.GaussianBlur(img, (blur, blur), 0)
        img += rng.normal(0, 3, img.shape)

        if SEDIMENT_CLOUD[0] <= t < SEDIMENT_CLOUD[1]:
            img = 0.4 * img + 0.6 * np.array([110, 150, 165])
        if BLOCKED[0] <= t < BLOCKED[1]:
            img *= 0.06  # lens covered / camera in the mud

        writer.write(np.clip(img, 0, 255).astype(np.uint8))
    writer.release()

    with open(TRUTH, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["start_s", "end_s", "n_fish"])
        for start, end, n, _, _ in FISH:
            out.writerow([start, end, n])
    print(f"Made {OUT.relative_to(ROOT)} ({SECONDS} s) and {TRUTH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
