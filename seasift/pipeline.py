"""Process one survey video from start to finish.

From the command line:
    python -m seasift.pipeline data/videos/test_murky.mp4
    python -m seasift.pipeline my_clip.mp4 --site coastal_eelgrass_example --ai
    python -m seasift.pipeline --all          # every new video in data/videos, one after another
"""
import argparse
import datetime
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

from . import config, db
from .energy import EnergyMeter, describe_kwh
from .species import decide_flag, load_species_list, site_csv
from .triage import MotionTriage, build_events
from .video import FrameReader, cached_frame, sample_frames, video_info
from .visibility import frame_visibility, minutes_by_label, summarize_windows


def scan_video(path, progress=None, boxes=None):
    """One pass over the video: motion triage + visibility for every sampled frame.

    boxes: pass an empty dict to also get where the movement was: {time_s: [(x, y, w, h), ...]}
    """
    fps, _, duration = video_info(path)
    triage = MotionTriage()
    motion_samples, vis_samples = [], []
    for t, frame in sample_frames(path, 1.0 / config.TRIAGE_FPS, width=config.PROCESS_WIDTH):
        active, motion = triage.step(frame)
        score, blocked = frame_visibility(frame)
        motion_samples.append((t, active, motion))
        vis_samples.append((t, score, blocked))
        if boxes is not None and triage.boxes:
            boxes[t] = triage.boxes
        if progress and duration and len(motion_samples) % 25 == 0:
            progress(min(0.9, 0.9 * t / duration), f"Scanning... {t / 60:.1f} of {duration / 60:.1f} min")
    if vis_samples:
        duration = max(duration, vis_samples[-1][0])
    return motion_samples, vis_samples, fps, duration


def review_moments(start_s, end_s, peak_s):
    """The pictures a reviewer sees for one event: the busiest moment, then three spread across it."""
    return [round(float(peak_s), 2)] + [round(float(start_s + (end_s - start_s) * f), 2) for f in (0.25, 0.5, 0.75)]


def thumb_folder(video_id):
    return config.THUMB_DIR / f"video_{video_id}"


def save_thumbnails(path, video_id, events, progress=None):
    """Save every picture the Review page will show, so reviewing never waits for the video."""
    folder = thumb_folder(video_id)
    with FrameReader(path) as reader:
        for i, ev in enumerate(events):
            times = review_moments(ev["start_s"], ev["end_s"], ev["peak_s"])
            for t in sorted(times):
                cached_frame(path, t, folder, width=config.REVIEW_WIDTH, reader=reader)
            peak = folder / f"t_{times[0]:08.2f}_w{config.REVIEW_WIDTH}.jpg"
            ev["thumb_path"] = config.stored_path(peak) if peak.exists() else None
            if progress and i % 5 == 0:
                progress(0.9 + 0.08 * i / max(1, len(events)), f"Saving review pictures ({i + 1} of {len(events)})")


def camera_handling(ev, duration):
    """Is this moment probably the camera being put in or taken out of the water?

    Coastal Action's protocol skips the first and last minute for this reason. Big movement
    a little later (or earlier) than that is almost always the camera being handled too.
    Short clips cut from a longer video have no handling, so they're never skipped.
    """
    if duration < config.CAMERA_MIN_VIDEO_S:
        return False
    t = ev["peak_s"]
    if t < config.SKIP_START_S or t > duration - config.SKIP_END_S:
        return True
    near_an_end = t < config.CAMERA_WINDOW_S or t > duration - config.CAMERA_WINDOW_S
    return near_an_end and ev["motion"] >= config.CAMERA_MOTION


def mark_camera_handling(conn, video_id, duration):
    """For videos processed before this check existed: mark their camera-handling moments."""
    events = db.query(conn, "SELECT * FROM events WHERE video_id = ? AND status = 'pending'", (video_id,))
    for ev in events.to_dict("records"):
        if camera_handling(ev, duration):
            db.set_event_status(conn, int(ev["id"]), "camera")


def detection_times(ev):
    """Which moments inside an event the AI looks at (every 1/DETECT_FPS seconds + the peak)."""
    times = list(np.arange(ev["start_s"], ev["end_s"], 1.0 / config.DETECT_FPS))
    if len(times) > config.MAX_DETECT_FRAMES_PER_EVENT:
        picks = np.linspace(0, len(times) - 1, config.MAX_DETECT_FRAMES_PER_EVENT).round().astype(int)
        times = [times[i] for i in picks]
    times.append(ev["peak_s"])
    return sorted({round(float(t), 2) for t in times})


def crop_with_margin(frame, x1, y1, x2, y2, margin=0.1):
    h, w = frame.shape[:2]
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    return frame[y1:y2, x1:x2]


def run_ai_tier(conn, video_id, path, events, species_df, progress=None):
    """Tier 1: detect fish inside events only, then suggest species and flag them."""
    from .ai import FishDetector, SpeciesSuggester  # heavy imports only when AI is used

    detector = FishDetector()
    suggester = SpeciesSuggester(species_df) if config.SPECIES_SUGGESTIONS else None
    folder = config.CROP_DIR / f"video_{video_id}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    with FrameReader(path) as reader:
        for i, ev in enumerate(events):
            if progress:
                progress(0.9 + 0.1 * i / max(1, len(events)), f"AI: event {i + 1} of {len(events)}")
            for t in detection_times(ev):
                frame = reader.at(t)
                if frame is None:
                    continue
                for (x1, y1, x2, y2, conf) in detector.detect(frame):
                    crop = crop_with_margin(frame, x1, y1, x2, y2)
                    if crop.size == 0:
                        continue
                    crop_path = folder / f"event{ev['id']}_t{t:.1f}_{len(rows)}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    top = (suggester.suggest(crop_path) if suggester else []) + [("", 0.0)] * 3
                    if suggester:
                        flag = decide_flag(top[0][0], top[0][1], ev["visibility"], species_df)
                    else:  # detector only: a person picks the species
                        flag = "unidentifiable" if ev["visibility"] < config.MURKY_SCORE else "fish"
                    rows.append({
                        "event_id": ev["id"], "video_id": video_id, "frame_s": t,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2, "det_conf": conf,
                        "crop_path": config.stored_path(crop_path),
                        "species_1": top[0][0], "score_1": top[0][1],
                        "species_2": top[1][0], "score_2": top[1][1],
                        "species_3": top[2][0], "score_3": top[2][1],
                        "flag": flag,
                    })
    db.add_detections(conn, rows)
    return rows


def recorded_date(path):
    """Best guess at the survey day: the date the video file was last changed (YYYY-MM-DD)."""
    return datetime.date.fromtimestamp(Path(path).stat().st_mtime).isoformat()


def process_video(path, site=None, use_ai=False, progress=None, conn=None, survey_date=None):
    """Run the whole pipeline on one video and save everything to the database.

    site: species list name (file name without .csv) or a path to the CSV.
    survey_date: the day the footage was recorded (YYYY-MM-DD). Defaults to the file's date.
    Returns (video_id, summary dict).
    """
    path = Path(path)
    survey_date = str(survey_date) if survey_date else recorded_date(path)
    conn = conn or db.connect()
    site_path = Path(site) if site and Path(site).exists() else (site_csv(site) if site else None)
    site_name = site_path.stem if site_path else ""
    if use_ai and site_path is None:
        raise ValueError("The AI tier needs a site species list.")

    detections = []
    with EnergyMeter() as meter:
        moved_at = {}
        motion, vis, fps, duration = scan_video(path, progress, boxes=moved_at)
        segments = summarize_windows(vis, duration)
        events = build_events(motion)
        for ev in events:
            ev["end_s"] = min(ev["end_s"], duration)
            ev["boxes"] = json.dumps(moved_at.get(ev["peak_s"], []))  # where it moved at the peak
            inside = [s for t, s, _ in vis if ev["start_s"] <= t <= ev["end_s"]]
            ev["visibility"] = round(statistics.median(inside), 1) if inside else 0.0
            ev["status"] = "camera" if camera_handling(ev, duration) else "pending"

        video_id = db.add_video(conn, path, site_name, duration, fps, use_ai, survey_date)
        to_review = [ev for ev in events if ev["status"] == "pending"]
        save_thumbnails(path, video_id, events, progress)
        db.add_segments(conn, video_id, segments)
        db.add_events(conn, video_id, events)
        if use_ai:
            detections = run_ai_tier(conn, video_id, path, to_review,
                                     load_species_list(site_path), progress)
    db.set_energy(conn, video_id, meter.kwh)
    if progress:
        progress(1.0, "Done")

    event_seconds = sum(ev["end_s"] - ev["start_s"] for ev in to_review)
    flags = {}
    for d in detections:
        flags[d["flag"]] = flags.get(d["flag"], 0) + 1
    summary = {
        "video_id": video_id,
        "duration_min": round(duration / 60, 1),
        "events": len(to_review),
        "camera_events": len(events) - len(to_review),
        "event_min": round(event_seconds / 60, 1),
        "skipped_share": round(1 - event_seconds / duration, 3) if duration else 0.0,
        "visibility_min": minutes_by_label(segments),
        "energy_kwh": meter.kwh,
        "ai_used": use_ai,
        "detections": len(detections),
        "flags": flags,
    }
    return video_id, summary


def unprocessed_videos(conn):
    """Videos in data/videos that haven't been processed yet (oldest name first)."""
    done = {row[0] for row in conn.execute("SELECT path FROM videos")}
    config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in config.VIDEO_DIR.iterdir() if p.suffix.lower() in config.VIDEO_TYPES)
    return [p for p in files if str(p.resolve()) not in done]


def print_summary(s):
    v = s["visibility_min"]
    print(f"\nVideo #{s['video_id']}: {s['duration_min']} min")
    print(f"  Events to review:  {s['events']} ({s['event_min']} min)"
          + (f", plus {s['camera_events']} camera handling (skipped)" if s.get("camera_events") else ""))
    print(f"  Footage skipped:   {s['skipped_share']:.0%}")
    print(f"  Visibility (min):  clear {v['clear']}, murky {v['murky']}, "
          f"unusable {v['unusable']}, blocked {v['blocked']}")
    print(f"  Energy used:       {describe_kwh(s['energy_kwh'])}")
    if s["ai_used"]:
        print(f"  AI detections:     {s['detections']}  flags: {s['flags']}")


def main():
    parser = argparse.ArgumentParser(description="Process survey videos with SeaSift.")
    parser.add_argument("video", nargs="?", help="one video file (or use --all)")
    parser.add_argument("--all", action="store_true",
                        help="process every new video in data/videos, one after another")
    parser.add_argument("--site", default="coastal_eelgrass_example",
                        help="species list name in data/species_lists (without .csv)")
    parser.add_argument("--ai", action="store_true", help="also run the local AI tier")
    parser.add_argument("--date", help="day the footage was recorded, YYYY-MM-DD (default: the file's date)")
    args = parser.parse_args()
    if not args.video and not args.all:
        parser.error("give a video file, or --all")
    videos = unprocessed_videos(db.connect()) if args.all else [args.video]
    if not videos:
        print("No new videos in data/videos.")
    for i, video in enumerate(videos, 1):
        print(f"\n[{i} of {len(videos)}] {Path(video).name}")
        _, summary = process_video(video, site=args.site, use_ai=args.ai, survey_date=args.date,
                                   progress=lambda f, m: print(f"\r{m:<60}", end="", flush=True))
        print_summary(summary)


if __name__ == "__main__":
    main()
