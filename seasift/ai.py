"""Tier 1: optional local AI (trained open-source models). Runs on this computer, offline.

  FishDetector      -> Community Fish Detector (YOLO, via ultralytics, AGPL-3.0). Finds fish boxes.
  SpeciesSuggester  -> BioCLIP (via pybioclip, MIT). Zero-shot species guesses from the site list.
                       Suggestions only: a person confirms every species. Turn it off with
                       config.SPECIES_SUGGESTIONS = False if it tests badly on your footage.

Heavy libraries are imported inside the classes, so the rest of SeaSift works
even if ultralytics / pybioclip are not installed.
Before going offline: run `python tools/check_ai.py` once so the models download.
"""
import importlib.util

from . import config


def ai_status():
    """Can the AI tier run here? Returns {"ready": bool, "problems": [what to fix]}."""
    problems = []
    if importlib.util.find_spec("ultralytics") is None:
        problems.append("Install the detector library: pip install -r requirements-ai.txt")
    if config.SPECIES_SUGGESTIONS and importlib.util.find_spec("bioclip") is None:
        problems.append("Install BioCLIP: pip install -r requirements-ai.txt "
                        "(or set SPECIES_SUGGESTIONS = False for detector only)")
    if not config.DETECTOR_WEIGHTS.exists():
        problems.append(f"Download the fish detector weights to {config.stored_path(config.DETECTOR_WEIGHTS)}")
    return {"ready": not problems, "problems": problems}


def pick_device():
    """Use the GPU if there is one: 'cuda' (NVIDIA), 'mps' (Apple Silicon Mac) or 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class FishDetector:
    """Finds fish in a frame. Returns a list of (x1, y1, x2, y2, confidence) in pixels."""

    def __init__(self, weights=config.DETECTOR_WEIGHTS, device=None):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.device = device or pick_device()

    def _predict(self, frame):
        return self.model.predict(frame, imgsz=config.DETECT_IMGSZ, conf=config.DETECT_CONF,
                                  device=self.device, verbose=False)[0]

    def detect(self, frame):
        try:
            result = self._predict(frame)
        except Exception:
            if self.device == "cpu":
                raise
            self.device = "cpu"          # some GPU setups fail on some layers: fall back to CPU
            result = self._predict(frame)
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        return [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
                for b, c in zip(xyxy, conf)]


class SpeciesSuggester:
    """Zero-shot species suggestions: BioCLIP picks from the names in the site's species list."""

    def __init__(self, species_df, device=None):
        import bioclip
        self.labels = list(species_df["scientific_name"])
        kwargs = {"cls_ary": self.labels, "device": device or pick_device()}
        if config.USE_BIOCLIP_V1 and hasattr(bioclip, "BIOCLIP_V1_MODEL_STR"):
            kwargs["model_str"] = bioclip.BIOCLIP_V1_MODEL_STR   # smaller, faster original BioCLIP
        try:
            self.classifier = bioclip.CustomLabelsClassifier(**kwargs)
        except TypeError:
            kwargs.pop("device")         # older pybioclip without a device option
            self.classifier = bioclip.CustomLabelsClassifier(**kwargs)
        except Exception:
            kwargs["device"] = "cpu"     # fall back to CPU if the GPU setup fails
            self.classifier = bioclip.CustomLabelsClassifier(**kwargs)

    def suggest(self, image_path, k=3):
        """Top-k (scientific_name, score) guesses for one crop image."""
        predictions = self.classifier.predict(str(image_path))
        predictions = sorted(predictions, key=lambda p: p["score"], reverse=True)[:k]
        return [(p["classification"], float(p["score"])) for p in predictions]
