"""Every number you might want to tune lives here, so nobody has to dig through the code.

Tip: tune the triage and visibility numbers on REAL Coastal Action clips,
one clear and one murky, and write down what you changed and why.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- folders
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"          # put survey videos here
SPECIES_DIR = DATA_DIR / "species_lists" # one CSV per survey site
MODEL_DIR = ROOT / "models"              # AI model weights (AI tier only)
# Results: the database, event pictures and fish crops. The smoke test points this
# somewhere else (SEASEARCH_OUT) so it never touches your real results.
OUT_DIR = Path(os.environ.get("SEASEARCH_OUT", DATA_DIR))
DB_PATH = OUT_DIR / "seasearch.db"
OLD_DB_PATH = OUT_DIR / "seasift.db"  # from before the app was renamed: picked up automatically
THUMB_DIR = OUT_DIR / "thumbs"   # review pictures, saved once so the Review page never waits for the video
COUNT_FRAME_DIR = OUT_DIR / "count_frames"  # the frames people count on (Count frames page)
CROP_DIR = OUT_DIR / "crops"
CLIP_DIR = OUT_DIR / "clips"   # enhanced event clips made by the Review page
LABEL_PHOTO_DIR = OUT_DIR / "label_photos"         # proof pictures saved with each confirmed species
REFERENCE_PHOTO_DIR = DATA_DIR / "species_photos"  # free-to-use photos for the species cards (with credit)
DEMO_DIR = OUT_DIR / "demo"    # side-by-side clips made by tools/make_demo_clip.py

VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def stored_path(path):
    """How a file path is saved in the database: relative to the project folder when the
    file is inside it (so the folder can move between laptops), otherwise the full path
    (e.g. results on an external drive)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def full_path(stored):
    """Turn a path from the database back into a real file path."""
    path = Path(stored)
    return path if path.is_absolute() else ROOT / path

# ---------------------------------------------------------------- reading video
PROCESS_WIDTH = 640    # frames are shrunk to this width before analysis (faster)
HW_DECODE = True       # ask for hardware video decoding (much faster for 4K on Macs); falls back
                       # to normal decoding automatically if the computer doesn't support it
TRIAGE_FPS = 4         # how many frames per second motion triage looks at

# ---------------------------------------------------------------- motion triage
WARMUP_S = 2.0          # ignore the first seconds while the background model learns
BG_HISTORY = 150        # how many sampled frames the background model remembers
BG_THRESHOLD = 25       # how different a pixel must be to count as "moved" (lower = more sensitive,
                        # catches fish in murkier water but also more noise)
MIN_BLOB_FRAC = 0.0008  # ignore moving blobs smaller than this share of the frame (sediment specks)
MAX_FG_FRAC = 0.40      # if more than this share of the frame changes at once, it's a camera
                        # bump or sediment cloud, not a fish
GLOBAL_CHANGE = 0.08    # if overall brightness or contrast changes more than 8% within ~1 s,
                        # the water/light is changing: re-learn the background, ignore motion
CHRONIC_ALPHA = 0.03    # how fast the "always moving" map learns (swaying grass, kelp)
CHRONIC_LEVEL = 0.35    # pixels that move more often than this count as "always moving"
MAX_CHRONIC_SHARE = 0.5 # blobs that are mostly inside "always moving" pixels are ignored
PERSIST_WINDOW = 5      # look at the last N sampled frames...
PERSIST_MIN = 3         # ...and need real motion in at least this many of them
MERGE_GAP_S = 2.0       # two events closer than this become one event
MIN_EVENT_S = 0.5       # events shorter than this are dropped
PAD_S = 1.0             # seconds added before and after every event

# ---------------------------------------------------------------- visibility
VIS_WINDOW_S = 10                # visibility is summarised in windows of this many seconds
CONTRAST_RANGE = (4.0, 40.0)     # grayscale spread mapped to 0..1
SHARPNESS_RANGE = (5.0, 400.0)   # edge strength mapped to 0..1 (log scale)
CLEAR_SCORE = 50                 # score at or above this = "clear"
MURKY_SCORE = 25                 # at or above this = "murky", below = "unusable"
DARK_LEVEL = 15                  # average brightness below this = lens blocked / too dark
BRIGHT_LEVEL = 240               # above this = blown out

# ---------------------------------------------------------------- enhancement (for reviewers' eyes)
CAST_STRENGTH = 0.7     # how much of the water's colour cast to remove (1 = all of it, which
                        # turned blue water pink on real photos; lower keeps more natural colour)
CLAHE_CLIP = 2.0        # local contrast boost (higher = punchier, but also grainier)

# ---------------------------------------------------------------- enhanced clips
CLIP_FPS = 12           # frames per second in enhanced event clips (lower = made faster)
MAX_CLIP_S = 30         # enhanced event clips stop after this many seconds
CLIP_LEAD_S = 5.0       # seconds watched before a clip starts, so the movement boxes are ready

# ---------------------------------------------------------------- camera handling (skipped automatically)
CAMERA_MIN_VIDEO_S = 5 * 60  # only for real deployments: a short clip cut from a video has no handling
CAMERA_WINDOW_S = 180   # big movement this close to the start or end is the camera being handled...
CAMERA_MOTION = 0.05    # ...when at least this share of the frame moved (moments in the first and
                        # last minute always count, like in Coastal Action's protocol)

# ---------------------------------------------------------------- reviewing help (no AI)
REVIEW_WIDTH = 960      # review pictures are saved this wide
MANY_FISH = 3           # more moving things than this at once: counting needs an expert
TINY_FISH_SHARE = 0.002 # biggest moving thing smaller than this share of the frame: too far to identify
NOT_SEEN_DAYS = 3 * 365 # a usual species not confirmed for this long shows under "not seen lately"

# ---------------------------------------------------------------- protocol mode (Coastal Action's MeanCount)
SKIP_START_S = 60        # skip the first minute: sediment from setting the camera down
SKIP_END_S = 60          # and the last minute
MAX_EXTRA_SKIP_S = 240   # skip up to 4 more minutes while the footage is still settling
LONG_VIDEO_S = 25 * 60   # ...or up to 10 more for videos longer than this (protocol comment)
MAX_EXTRA_SKIP_LONG_S = 600
VERY_LONG_VIDEO_S = 30 * 60  # ...or up to 15 more for videos longer than this (protocol comment)
MAX_EXTRA_SKIP_VERY_LONG_S = 900
SETTLING_MOTION = 0.20   # a moment with this much of the frame moving = still settling / being set up
SUBSAMPLE_EVERY_S = 60   # one sub-sample frame every 60 seconds
SUBSAMPLE_FRAMES = 15    # 15 frames = a 15-minute analysis window
CONTEXT_S = 5            # watch this many seconds before and after each frame
CONTEXT_SPEED = 0.25     # ...at quarter speed
GRID_ROWS = 3            # counting grid like Coastal Action's: square cells, a third of the frame high
                         # (on 16:9 video that is 5 full columns and a narrow 6th)
COUNT_WIDTH = 1600       # frames are shown this wide for counting (bigger = easier to see small fish)
DOT_RADIUS = 0.006       # clicking this close to a dot (share of the frame height) removes it
QC_SHARE = 0.10          # quality check: at least 10% of frames counted again, blind...
QC_MIN_FRAMES = 6        # ...and at least this many (the protocol says 6-12 per video)
QC_MAX_DIFF_PCT = 6.0    # the two counts should differ by less than this percent

# ---------------------------------------------------------------- AI tier (optional)
DETECTOR_WEIGHTS = MODEL_DIR / "cfd-yolov12x-1.00.pt"  # Community Fish Detector weights
AI_DEVICE = os.environ.get("SEASEARCH_AI_DEVICE", "")  # "" = pick automatically: cuda, then mps (macOS 15+
                        # only, it crashes on older macOS), then cpu. Or force one, e.g. SEASEARCH_AI_DEVICE=cpu
DETECT_IMGSZ = 640      # smaller = faster on a laptop (the model was trained at 1024)
DETECT_CONF = 0.25      # ignore fish boxes below this confidence
SCENERY_IOU = 0.5       # two AI boxes overlapping this much (0-1) are in the same spot
SCENERY_MOMENTS = 2     # an AI "fish" in the same spot in this many different moments is scenery
                        # (bands on a measuring pole, a rock): real fish move. Those boxes are hidden.
DETECT_FPS = 0.5        # inside an event, check one frame every 2 seconds
MAX_DETECT_FRAMES_PER_EVENT = 10
SPECIES_SUGGESTIONS = True  # False = detector only: the AI draws fish boxes and counts them,
                            # a person picks every species. Use this if BioCLIP tests badly.
USE_BIOCLIP_V1 = False  # False = BioCLIP 2 (better: on real Coastal Action footage it named all 8 crabs and
                        # shrimp right, BioCLIP 1 got 0-2; about 0.4 s per crop on a laptop CPU).
                        # True = the original BioCLIP (smaller, about 0.1 s per crop)
LOOK_AT_MOVEMENT = True # also ask BioCLIP about what moved in each moment: finds crabs and shrimp,
                        # which the fish detector never boxes
SPECIES_MIN_SCORE = 0.35  # species guesses below this score are marked "unsure"
