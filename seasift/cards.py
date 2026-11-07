"""Help a person identify a fish: three quick questions and a card for each species.

No AI. The questions only hide the species whose features don't match the answers,
like a paper field guide that does the page-flipping for you. The person picks the card.
Features come from the site's species list (columns body, tail, markings, tip, photo).
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import config, db, species

WHAT = {"fish": "Fish", "invertebrate": "Crab, shrimp or other invertebrate"}  # the first question
QUESTIONS = {  # the fish questions
    "body": ["long and thin", "torpedo", "tall and round", "flat"],
    "tail": ["forked", "rounded", "square"],
    "markings": ["spots", "stripes", "plain"],
}
CANT_SEE = "can't see"


def feature_values(cell):
    """'forked|square' -> ['forked', 'square']. Blank -> [] (we don't know, so never hide it)."""
    return [v.strip() for v in str(cell or "").split("|") if v.strip()]


def matches(row, answers):
    """True if this species fits every answer. "can't see" and blank features never hide a species.

    answers: {"what": fish / invertebrate, "body": ..., "tail": ..., "markings": ...}
    """
    for feature, answer in answers.items():
        if not answer or answer == CANT_SEE:
            continue
        if feature == "what":
            if species.group_of(row) != answer:
                return False
            continue
        values = feature_values(row.get(feature, ""))
        if values and answer not in values:
            return False
    return True


def filter_species(species_df, answers):
    """Only the species that fit the answers."""
    keep = [matches(row, answers) for row in species_df.to_dict("records")]
    return species_df[keep]


def describe(answers):
    """The answers as text to save with the label, e.g. 'fish, body: torpedo, tail: forked'."""
    seen = [answer if feature == "what" else f"{feature}: {answer}"
            for feature, answer in answers.items() if answer and answer != CANT_SEE]
    return ", ".join(seen)


def describe_features(row):
    """What a species looks like, in words: 'torpedo body, forked tail, spots'."""
    words = {"body": "{} body", "tail": "{} tail", "markings": "{}"}
    parts = [words[f].format(" or ".join(feature_values(row.get(f, "")))) for f in QUESTIONS
             if feature_values(row.get(f, ""))]
    return ", ".join(parts)


def card_order(species_df, seen_counts, ai_scores=None):
    """Card order: the AI's suggestions first (if it ran), then the species seen most at this site."""
    ai_scores = ai_scores or {}

    def key(row):
        name = row["scientific_name"]
        return (name not in ai_scores, -ai_scores.get(name, 0.0), -seen_counts.get(name, 0),
                row["common_name"])

    rows = sorted(species_df.to_dict("records"), key=key)
    return [row["scientific_name"] for row in rows]


def ai_scores_for_event(detections):
    """Best AI score per species in one event's detections: {scientific name: score}."""
    scores = {}
    for row in detections.to_dict("records"):
        for k in (1, 2, 3):
            name, score = row.get(f"species_{k}"), row.get(f"score_{k}")
            if isinstance(name, str) and name and score == score and score is not None:
                scores[name] = max(scores.get(name, 0.0), float(score))
    return scores


def seen_counts_at_site(conn, site):
    """How many times each species was confirmed at this site: {scientific name: count}."""
    rows = db.query(conn, "SELECT l.species, COUNT(*) AS n FROM labels l JOIN videos v ON l.video_id = v.id "
                          "WHERE v.site = ? GROUP BY l.species", (site,))
    return dict(zip(rows.species, rows.n))


def own_photos(conn):
    """The clearest confirmed picture of each species from your own footage: {name: photo path}."""
    rows = db.query(conn, "SELECT l.species, l.photo_path FROM labels l JOIN events e ON l.event_id = e.id "
                          "WHERE l.photo_path IS NOT NULL ORDER BY e.visibility DESC, l.id DESC")
    photos = {}
    for name, path in zip(rows.species, rows.photo_path):
        if name not in photos and config.full_path(path).exists():
            photos[name] = path
    return photos


def card_photo(row, photos):
    """Best picture for a card: your own footage, then a photo file, then None.

    Returns (image path or None, caption)."""
    name = row["scientific_name"]
    if name in photos:
        return config.full_path(photos[name]), "From your own footage"
    if row.get("photo"):
        path = config.REFERENCE_PHOTO_DIR / row["photo"]
        if path.exists():
            credit = row.get("photo_credit") or ""
            return path, f"Photo: {credit}" if credit else ""
    return None, ""


@lru_cache(maxsize=64)
def outline(body, group="fish", width=320, height=180):
    """A simple drawing of the body shape, for species with no photo yet (BGR picture).

    Drawn once per shape and kept in memory. Don't draw on the picture it returns.
    """
    img = np.full((height, width, 3), 238, np.uint8)
    cv2.putText(img, "No photo yet", (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1,
                cv2.LINE_AA)
    if group == "invertebrate":
        return _crab(img, width // 2, height // 2 - 10)
    shapes = {"long and thin": (120, 12), "torpedo": (90, 28), "tall and round": (62, 48), "flat": (85, 52)}
    first = feature_values(body)[0] if feature_values(body) else ""
    if first not in shapes:  # body shape unknown (or not a fish, like a crab): no outline
        return img
    ax, ay = shapes[first]
    cx, cy = width // 2 - 18, height // 2 - 8
    grey = (165, 165, 165)
    cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, grey, -1)
    tail = np.array([(cx + ax - 6, cy), (cx + ax + 32, cy - max(ay, 16)), (cx + ax + 32, cy + max(ay, 16))])
    cv2.fillPoly(img, [tail], grey)
    cv2.circle(img, (cx - ax + max(ax // 6, 10), cy - ay // 4), 4, (95, 95, 95), -1)
    return img


def _crab(img, cx, cy):
    """A plain crab shape: a shell, four legs on each side and two claws."""
    grey, dark = (165, 165, 165), (95, 95, 95)
    for side in (-1, 1):
        for k in range(4):
            start = (cx + side * 42, cy - 8 + k * 9)
            knee = (cx + side * (74 + k * 2), cy - 20 + k * 15)
            cv2.line(img, start, knee, grey, 5, cv2.LINE_AA)
            cv2.line(img, knee, (cx + side * (88 + k * 3), cy + 6 + k * 15), grey, 4, cv2.LINE_AA)
        cv2.line(img, (cx + side * 30, cy - 18), (cx + side * 46, cy - 40), grey, 6, cv2.LINE_AA)
        cv2.ellipse(img, (cx + side * 50, cy - 50), (13, 9), side * 30, 0, 360, grey, -1)
    cv2.ellipse(img, (cx, cy), (52, 32), 0, 0, 360, grey, -1)
    for side in (-1, 1):
        cv2.circle(img, (cx + side * 12, cy - 26), 4, dark, -1)
    return img


def save_label_photo(frame, label_id):
    """Save a small copy of the proof picture for the species cards. Returns the stored path."""
    config.LABEL_PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    path = config.LABEL_PHOTO_DIR / f"label_{label_id}.jpg"
    if frame.shape[1] > 480:
        frame = cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1])))
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return config.stored_path(Path(path))
