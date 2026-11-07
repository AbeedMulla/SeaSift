"""Measure how well motion triage works against known fish times. These are your "receipts".

With a simple answer-key CSV (start_s,end_s[,n_fish]):
    python tools/eval_triage.py data/videos/test_murky.mp4 --truth data/videos/test_murky_truth.csv

With Brackish-style YOLO label files (one .txt per frame, frame number at the end of the name):
    python tools/eval_triage.py VIDEO.avi --yolo-dir path/to/labels --fps 15 --fish-classes 0,4

Prints: fish passages found, fish-time covered, footage skipped, false-alarm events, true MaxN.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seasift.pipeline import scan_video  # noqa: E402
from seasift.triage import build_events  # noqa: E402
from seasift.visibility import label_for  # noqa: E402


def truth_from_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    intervals = [(float(r["start_s"]), float(r["end_s"])) for r in rows]
    maxn = max((int(r.get("n_fish") or 1) for r in rows), default=0)
    return intervals, maxn


def truth_from_yolo(folder, fps, fish_classes, gap_s=0.5):
    """Frames with at least one fish box -> time intervals. Also returns true MaxN (most boxes in a frame)."""
    fish_frames, maxn = {}, 0
    for txt in Path(folder).glob("*.txt"):
        numbers = re.findall(r"\d+", txt.stem)
        if not numbers:
            continue
        frame = int(numbers[-1])
        n = sum(1 for line in txt.read_text().splitlines()
                if line.strip() and int(line.split()[0]) in fish_classes)
        if n:
            fish_frames[frame] = n
            maxn = max(maxn, n)
    times = sorted(f / fps for f in fish_frames)
    intervals = []
    for t in times:
        if intervals and t - intervals[-1][1] <= gap_s:
            intervals[-1][1] = t
        else:
            intervals.append([t, t])
    return [(a, b + 1 / fps) for a, b in intervals], maxn


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--truth", help="CSV with start_s,end_s[,n_fish]")
    p.add_argument("--yolo-dir")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--fish-classes", default="0", help="comma-separated YOLO class ids that are fish")
    args = p.parse_args()

    if args.truth:
        truth, true_maxn = truth_from_csv(args.truth)
    elif args.yolo_dir:
        classes = {int(c) for c in args.fish_classes.split(",")}
        truth, true_maxn = truth_from_yolo(args.yolo_dir, args.fps, classes)
    else:
        p.error("give --truth or --yolo-dir")

    motion, vis, _, duration = scan_video(args.video)
    events = [(e["start_s"], min(e["end_s"], duration)) for e in build_events(motion)]

    found = [any(overlap(t, e) > 0 for e in events) for t in truth]
    fish_time = sum(b - a for a, b in truth)
    covered = sum(min(b - a, sum(overlap((a, b), e) for e in events)) for a, b in truth)
    event_time = sum(b - a for a, b in events)
    false_alarms = sum(1 for e in events if not any(overlap(t, e) > 0 for t in truth))

    print(f"\nVideo: {args.video} ({duration / 60:.1f} min)")
    print(f"Fish passages found:   {sum(found)} of {len(truth)}")
    for (a, b), ok in zip(truth, found):
        scores = [s for t, s, _ in vis if a <= t <= b]
        water = label_for(sorted(scores)[len(scores) // 2]) if scores else "?"
        print(f"   {a:7.1f}-{b:7.1f} s  {'found ' if ok else 'MISSED'}  (water: {water})")
    print(f"Fish time covered:     {covered / fish_time:.0%}" if fish_time else "")
    print(f"Footage skipped:       {1 - event_time / duration:.0%}")
    print(f"Events:                {len(events)} ({false_alarms} with no fish)")
    print(f"True MaxN (answer key): {true_maxn}")
    print(f"\nSlide line: skipped {1 - event_time / duration:.0%} of footage and found "
          f"{sum(found)} of {len(truth)} fish passages.")


if __name__ == "__main__":
    main()
