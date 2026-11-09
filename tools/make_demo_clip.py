"""Make a side-by-side clip for the pitch: the original on the left, seaSEArch's enhanced version
on the right, with orange boxes where it saw movement and the footage-quality score.

    python tools/make_demo_clip.py data/videos/v06.mp4 --start 5 --end 20

Saves to data/demo/. Classical computer vision only (no trained model), the same as the app.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seasearch import clips, config  # noqa: E402
from seasearch.video import video_info  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Original vs enhanced, side by side, for slides.")
    p.add_argument("video")
    p.add_argument("--start", type=float, default=0.0, help="seconds (default 0)")
    p.add_argument("--end", type=float, help="seconds (default: 30 s after the start, or the end)")
    p.add_argument("--no-boxes", action="store_true", help="don't draw the orange movement boxes")
    p.add_argument("--width", type=int, default=640, help="width of each half in pixels (default 640)")
    args = p.parse_args()

    _, _, duration = video_info(args.video)
    end = min(duration, args.end if args.end is not None else args.start + 30)
    if end <= args.start:
        sys.exit("--end must be after --start (and inside the video)")
    out_base = config.DEMO_DIR / f"{Path(args.video).stem}_{args.start:.0f}-{end:.0f}s_side_by_side"

    print(f"Making {end - args.start:.0f} s of side-by-side video...")
    path, kind = clips.make_clip(args.video, args.start, end, out_base, compare=True,
                                 boxes=not args.no_boxes, width=args.width, fps_out=30,
                                 formats=clips.FOR_SLIDES)
    if path is None:
        sys.exit("OpenCV on this computer can't write any of the video formats. Try: pip install -U opencv-python")
    print(f"Saved {path} ({kind})")
    if kind != "H.264 MP4":
        print("Slides prefer H.264. If you have ffmpeg, convert it with:\n"
              f"  ffmpeg -i {path} -c:v libx264 -pix_fmt yuv420p {path.with_name(path.stem + '_h264.mp4')}")


if __name__ == "__main__":
    main()
