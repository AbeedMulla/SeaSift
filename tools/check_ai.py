"""Day-1 check for the AI tier. Run this ONCE on the demo laptop while you have internet:
it downloads BioCLIP, then tells you how fast each model is on this machine.

Usage:  python tools/check_ai.py [video_or_image]   (default: data/videos/test_murky.mp4)

Before running: pip install -r requirements-ai.txt, and put the Community Fish Detector
weights (cfd-yolov12x-1.00.pt, from the project's GitHub Releases page) in models/.
"""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seasift import config  # noqa: E402
from seasift.ai import FishDetector, SpeciesSuggester, ai_status, pick_device  # noqa: E402
from seasift.species import load_species_list  # noqa: E402
from seasift.video import FrameReader  # noqa: E402


def load_test_frame(path):
    path = Path(path)
    if path.suffix.lower() in config.VIDEO_TYPES:
        with FrameReader(path) as reader:
            return reader.at(10.5)
    return cv2.imread(str(path))


def main():
    status = ai_status()
    print("AI tier ready:", status["ready"])
    for problem in status["problems"]:
        print("  -", problem)
    if not status["ready"]:
        return
    print("Device:", pick_device())

    source = sys.argv[1] if len(sys.argv) > 1 else config.VIDEO_DIR / "test_murky.mp4"
    frame = load_test_frame(source)
    if frame is None:
        print("Couldn't read a test frame from", source)
        return

    t = time.time()
    detector = FishDetector()
    print(f"Detector loaded in {time.time() - t:.1f} s")
    detector.detect(frame)  # warm-up
    t = time.time()
    for _ in range(3):
        boxes = detector.detect(frame)
    per_frame = (time.time() - t) / 3
    print(f"Detector: {per_frame:.2f} s per frame, found {len(boxes)} fish in the test frame")

    species_df = load_species_list(config.SPECIES_DIR / "coastal_eelgrass_example.csv")
    t = time.time()
    suggester = SpeciesSuggester(species_df)
    print(f"BioCLIP loaded in {time.time() - t:.1f} s (first run downloads it)")
    crop_path = config.DATA_DIR / "check_ai_crop.jpg"
    if boxes:
        x1, y1, x2, y2, _ = boxes[0]
        cv2.imwrite(str(crop_path), frame[int(y1):int(y2), int(x1):int(x2)])
    else:
        cv2.imwrite(str(crop_path), frame)
    suggester.suggest(crop_path)  # warm-up
    t = time.time()
    for _ in range(3):
        top = suggester.suggest(crop_path)
    per_crop = (time.time() - t) / 3
    print(f"BioCLIP: {per_crop:.2f} s per fish crop. Top guesses: {top}")
    print(f"\nWrite these down for the pitch: detector {per_frame:.2f} s/frame, "
          f"species {per_crop:.2f} s/crop on {pick_device()}.")


if __name__ == "__main__":
    main()
