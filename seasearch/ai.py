"""Tier 1: optional local AI (trained open-source models). Runs on this computer, offline.

  FishDetector      -> Community Fish Detector (YOLO, via ultralytics, AGPL-3.0). Finds fish boxes.
  SpeciesSuggester  -> BioCLIP (via pybioclip, MIT). Zero-shot species guesses from the site list.
                       Suggestions only: a person confirms every species. Turn it off with
                       config.SPECIES_SUGGESTIONS = False if it tests badly on your footage.

Heavy libraries are imported inside the classes, so the rest of seaSEArch works
even if ultralytics / pybioclip are not installed.
Before going offline: run `python tools/check_ai.py` once so the models download.
"""
import importlib.util
import platform

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


def mac_gpu_safe():
    """Newer PyTorch crashes the whole app on the Mac GPU before macOS 15 (it calls a Metal
    function older macOS doesn't have, so Python can't catch it). Use the CPU there instead."""
    try:
        return int(platform.mac_ver()[0].split(".")[0]) >= 15
    except ValueError:
        return False


def pick_device():
    """Use the GPU if there is one: 'cuda' (NVIDIA), 'mps' (Apple Silicon Mac) or 'cpu'."""
    if config.AI_DEVICE:
        return config.AI_DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available() and mac_gpu_safe():
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


# "What kind of thing is this?" BioCLIP picks one. Tested on real Coastal Action footage
# (Nov 2026): with BioCLIP 2 these named all 8 crabs and shrimp right and called most sand,
# water and seaweed "not an animal". BioCLIP 1 got 0-2 of the 8.
KIND_PROMPTS = {
    "a photo of a crab": "crab", "a photo of a shrimp": "shrimp", "a photo of a fish": "fish",
    "a photo of seaweed": "none", "a photo of sand": "none", "a photo of a rope": "none",
    "a photo of a plastic pipe": "none", "a photo of murky water": "none",
}
INVERTEBRATE_KINDS = {"crab", "shrimp"}


class SpeciesSuggester:
    """Zero-shot guesses with BioCLIP, from the names in the site's species list.

    kind(crop)    -> ("crab" / "shrimp" / "fish" / "none", score): what kind of thing it is
    suggest(crop) -> top species, only among fish or only among invertebrates when group is given
    One model is loaded and shared by every label list (BioCLIP 2 is big), and each list's text
    features are saved in models/bioclip_cache, so only the very first run spends time on them.
    """

    def __init__(self, species_df, device=None):
        import bioclip
        from .species import names_in_group
        model_str = bioclip.BIOCLIP_V1_MODEL_STR if config.USE_BIOCLIP_V1 else bioclip.BIOCLIP_MODEL_STR
        kwargs = {"cls_ary": list(KIND_PROMPTS), "device": device or pick_device(), "model_str": model_str}
        try:
            self.classifier = bioclip.CustomLabelsClassifier(**kwargs)
        except Exception:
            kwargs["device"] = "cpu"     # fall back to CPU if the GPU setup fails
            self.classifier = bioclip.CustomLabelsClassifier(**kwargs)
        self.model_str = model_str
        names = list(species_df["scientific_name"])
        inverts = names_in_group(species_df, "invertebrate")
        self.labels = {
            "kind": list(KIND_PROMPTS),
            "all": names,
            "fish": [n for n in names if n not in inverts] or names,  # fish, plus plants (seaweed = false alarm)
            "invertebrate": inverts or names,
        }
        self._features = {"kind": self.classifier.txt_embeddings}

    def _text_features(self, key):
        """Text features for one label list, from the cache on disk if we've made them before."""
        if key not in self._features:
            import hashlib
            import torch
            labels = self.labels[key]
            digest = hashlib.sha1("\n".join([self.model_str, *labels]).encode()).hexdigest()[:16]
            path = config.MODEL_DIR / "bioclip_cache" / f"{digest}.pt"
            device = self.classifier.device
            if path.exists():
                features = torch.load(path, map_location=device)
            else:
                features = self.classifier._get_txt_embeddings(labels)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(features.cpu(), path)
            self._features[key] = features.to(device)
        return self._features[key]

    def _predict(self, image_path, key):
        clf = self.classifier
        clf.classes, clf.txt_embeddings = self.labels[key], self._text_features(key)
        predictions = clf.predict(str(image_path))
        return sorted(predictions, key=lambda p: p["score"], reverse=True)

    def kind(self, image_path):
        """What kind of thing is in the crop: ("crab" / "shrimp" / "fish" / "none", score)."""
        top = self._predict(image_path, "kind")[0]
        return KIND_PROMPTS[top["classification"]], float(top["score"])

    def suggest(self, image_path, k=3, group="all"):
        """Top-k (scientific_name, score) guesses. group: "all", "fish" or "invertebrate"."""
        return [(p["classification"], float(p["score"])) for p in self._predict(image_path, group)[:k]]
