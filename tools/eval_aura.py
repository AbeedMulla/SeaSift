"""Test seaSEArch on real underwater video: the AURA dataset (Denmark, CC BY 4.0).

AURA has 25 real videos. Scene A is a harbour 11 m deep, where light, visibility and marine snow
change from clip to clip. Scene B is a camera on a bridge, 9 m deep, lit by artificial light, in brackish water.
16 people watched every video and marked when something was happening (fish, crabs, other life).
consensus_events.csv holds the average of their start and end frames: our answer key.

1. Download a few videos, plus consensus_events.csv and soft_labels.csv, from
   https://zenodo.org/records/18139275 into data/videos/
2. Run:  python tools/eval_aura.py            (or give another folder: python tools/eval_aura.py FOLDER)

Each video gets the same triage as the app (nothing is saved). For each one it prints:
did seaSEArch find the moment people marked, how much of that moment it kept, how much footage
it skipped, and how many extra moments it made where nobody marked anything.

Credit on slides: AURA dataset, Weihl, Bengtson, Novak & Pedersen (2025), ICCV Workshops. CC BY 4.0.
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seasearch import config  # noqa: E402
from seasearch.pipeline import scan_video  # noqa: E402
from seasearch.triage import build_events  # noqa: E402
from seasearch.video import video_info  # noqa: E402
from seasearch.visibility import label_for  # noqa: E402

DOWNLOAD_PAGE = "https://zenodo.org/records/18139275"
CREDIT = "AURA dataset: Weihl, Bengtson, Novak & Pedersen (2025), ICCV Workshops. CC BY 4.0."
PLACES = {"scene_A": "harbour", "scene_B": "bridge"}
EDGE_S = 1.0  # people disagree by about a second at the edges, so allow that much slack


def find_file(folder, name):
    """First file with this name anywhere inside the folder (the zip version uses subfolders)."""
    matches = sorted(Path(folder).rglob(name))
    return matches[0] if matches else None


def read_answer_key(path):
    """consensus_events.csv -> {"v00": ("scene_A", 225.0, 419.0), ...}"""
    with open(path, newline="") as f:
        return {row["video"]: (row["scene"], float(row["start_frame"]), float(row["end_frame"]))
                for row in csv.DictReader(f)}


def count_label_frames(path):
    """soft_labels.csv has one row per labelled frame -> {"v00": 438, ...}"""
    counts = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            counts[row["video"]] = counts.get(row["video"], 0) + 1
    return counts


def overlap(a, b):
    """Seconds that two (start, end) intervals share."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def score_video(path, start_frame, end_frame, label_frames=None):
    """Scan one video and compare seaSEArch's moments with the moment people marked."""
    fps, n_frames, video_s = video_info(path)

    # Frame number -> seconds. If the labels count a different number of frames than the
    # video has (frames were sampled), spread the labelled frames evenly over the video.
    scaled = bool(label_frames) and label_frames != n_frames
    seconds_per_frame = video_s / label_frames if scaled else 1.0 / fps
    marked = (start_frame * seconds_per_frame, (end_frame + 1) * seconds_per_frame)

    motion, vis, _, duration = scan_video(path)
    moments = [(e["start_s"], min(e["end_s"], duration)) for e in build_events(motion)]
    with_slack = (marked[0] - EDGE_S, marked[1] + EDGE_S)
    kept_s = sum(b - a for a, b in moments)

    scores = [s for t, s, _ in vis if marked[0] <= t <= marked[1]] or [s for _, s, _ in vis]
    return {
        "length_s": duration,
        "marked": marked,
        "found": any(overlap(marked, m) > 0 for m in moments),
        "covered": min(1.0, sum(overlap(marked, m) for m in moments) / max(marked[1] - marked[0], 1e-6)),
        "kept_s": kept_s,
        "skipped": max(0.0, 1 - kept_s / duration) if duration else 0.0,
        "extra": sum(1 for m in moments if overlap(with_slack, m) == 0),
        "water": label_for(statistics.median(scores)) if scores else "?",
        "scaled": scaled,
    }


def score_folder(folder, quiet=False):
    """Score every AURA video found in the folder. Returns one result dict per video."""
    key_path = find_file(folder, "consensus_events.csv")
    if key_path is None:
        raise FileNotFoundError(f"No consensus_events.csv in {folder}. Download it from {DOWNLOAD_PAGE}")
    answer_key = read_answer_key(key_path)
    soft_path = find_file(folder, "soft_labels.csv")
    label_counts = count_label_frames(soft_path) if soft_path else {}

    videos = sorted(p for p in Path(folder).rglob("*")
                    if p.suffix.lower() in config.VIDEO_TYPES and p.stem in answer_key)
    results = []
    for path in videos:
        scene, start_frame, end_frame = answer_key[path.stem]
        if not quiet:
            print(f"Scanning {path.name} ...", flush=True)
        result = score_video(path, start_frame, end_frame, label_counts.get(path.stem))
        result.update(video=path.stem, place=PLACES.get(scene, scene))
        results.append(result)
    return results


def totals(group):
    """(found, videos, share of footage skipped, extra moments) for a group of results."""
    length = sum(r["length_s"] for r in group)
    kept = sum(r["kept_s"] for r in group)
    return (sum(r["found"] for r in group), len(group),
            1 - kept / length if length else 0.0, sum(r["extra"] for r in group))


def print_results(results):
    print(f"\n{'Video':6} {'Place':8} {'Length':>6}  {'People marked':>14}  {'Found?':7}"
          f"{'Kept':>5} {'Skipped':>8} {'Extra':>6}  Water")
    for r in results:
        a, b = r["marked"]
        print(f"{r['video']:6} {r['place']:8} {r['length_s']:5.0f}s  {a:6.1f}-{b:5.1f} s  "
              f"{'yes' if r['found'] else 'MISSED':7}{r['covered']:5.0%} {r['skipped']:8.0%} "
              f"{r['extra']:6}  {r['water']}")

    print()
    for place in ["harbour", "bridge"]:
        group = [r for r in results if r["place"] == place]
        if group:
            found, n, skipped, extra = totals(group)
            print(f"{place.capitalize():8} found {found} of {n} marked moments, skipped {skipped:.0%} "
                  f"of the footage, {extra} extra moments")

    found, n, skipped, extra = totals(results)
    if any(r["scaled"] for r in results):
        print("\nNote: the labels count frames differently from the video, so times were scaled to fit.")
    print("Note: AURA clips are short and cut around the action, so 'skipped' is lower here than it"
          " would be on hours of survey footage where fish are rare.")
    print(f"\nSlide line: on {n} real underwater videos from Denmark, each marked by 16 people,"
          f" seaSEArch found {found} of {n} marked moments and skipped {skipped:.0%} of the footage.")
    print(f"Credit: {CREDIT}")


def main():
    p = argparse.ArgumentParser(description="Score seaSEArch on real AURA underwater videos.")
    p.add_argument("folder", nargs="?", default=str(config.VIDEO_DIR),
                   help="folder with AURA videos (v00.mp4 ...) and consensus_events.csv")
    args = p.parse_args()
    try:
        results = score_folder(args.folder)
    except FileNotFoundError as e:
        sys.exit(str(e))
    if not results:
        sys.exit(f"No AURA videos (v00.mp4 ... v24.mp4) in {args.folder}. Download some from {DOWNLOAD_PAGE}")
    print_results(results)


if __name__ == "__main__":
    main()
