"""Species lists and the review flags. Plain Python, no AI.

A species list is a CSV per site with these columns:
  common_name, scientific_name, status, expected
  status:   native / invasive / tropical_stray / at_risk / not_fish
  expected: yes / no   (would you normally see it at this site?)

Optional columns, used by the species cards on the Review page:
  group                 fish (or blank) / invertebrate (crabs, shrimp, lobster, snails...)
  body, tail, markings  what the fish looks like (see cards.QUESTIONS for the words to use;
                        several allowed with |, e.g. "forked|square"; blank = not sure)
  tip                   one short ID tip, in our own words
  photo, photo_credit   a free-to-use photo in data/species_photos/ and who took it
  first_seen, evidence  filled in when a new arrival is added from the footage

To add a new arrival (e.g. a tropical stray nobody expected), add a row.
The AI tier reads the list at run time, so no retraining is needed.
"""
from pathlib import Path

import pandas as pd

from . import config

STATUSES = {"native", "invasive", "tropical_stray", "at_risk", "not_fish"}
BASE_COLUMNS = ["common_name", "scientific_name", "status", "expected"]
EXTRA_COLUMNS = ["group", "body", "tail", "markings", "tip", "photo", "photo_credit", "first_seen", "evidence"]
GROUPS = ["fish", "invertebrate"]
WATCH_STATUSES = {"invasive", "tropical_stray"}

FLAG_TEXT = {
    "known": "Expected species",
    "unexpected": "Unexpected or watch-list species: get an expert to check",
    "unidentifiable": "Too murky to identify",
    "unsure": "AI is unsure",
    "false_alarm": "Probably not an animal (seaweed, sand, rope...), hidden",
    "fish": "Fish found (the reviewer picks the species)",
    "scenery": "Not a fish: in the same spot in several moments (pole, rock), hidden",
}


CATEGORY_TEXT = {
    "normal": "Normal here",
    "unusual": "Not usually seen here",
    "tropical": "Tropical stray",
    "invasive": "Invasive",
    "at_risk": "Species at risk",
    "new": "New: not on the list",
    "not_fish": "Not a fish",
}
CATEGORY_COLOR = {"normal": "green", "unusual": "orange", "tropical": "orange", "invasive": "red",
                  "at_risk": "violet", "new": "blue", "not_fish": "gray"}
ATTENTION = {"unusual", "tropical", "invasive", "at_risk", "new"}


def row_category(row):
    """The tag for one species, from its row on the site list (a dict or a pandas row)."""
    by_status = {"tropical_stray": "tropical", "invasive": "invasive",
                 "at_risk": "at_risk", "not_fish": "not_fish"}
    if row["status"] in by_status:
        return by_status[row["status"]]
    return "normal" if row["expected"] else "unusual"


def species_category(species_df, name):
    """Which tag a confirmed species gets at this site. No AI: just the site's species list.

    normal   = native and usually found here      invasive = on the list as invasive
    unusual  = native but not usually found here  at_risk  = species at risk (share its location carefully)
    tropical = tropical stray (a climate signal)  new      = not on this site's list at all
    """
    if species_df is None or not name:
        return "new"
    row = species_df[species_df["scientific_name"] == name]
    return row_category(row.iloc[0]) if not row.empty else "new"


def group_of(row):
    """fish / invertebrate / plant, for one row of a species list (a dict). Blank group = fish."""
    if row.get("status") == "not_fish":
        return "plant"
    return "invertebrate" if str(row.get("group", "")).strip().lower() == "invertebrate" else "fish"


def names_in_group(species_df, group):
    """Scientific names of the fish (or invertebrates) on a site list."""
    if species_df is None:
        return []
    return [r["scientific_name"] for r in species_df.to_dict("records") if group_of(r) == group]


def badge(category):
    """A coloured tag for Streamlit markdown, e.g. :red-badge[Invasive]."""
    return f":{CATEGORY_COLOR[category]}-badge[{CATEGORY_TEXT[category]}]"


def is_unexpected(species_df, name):
    """True for species a person confirmed that aren't on the list, aren't usually here,
    or are invasive / tropical strays."""
    return species_category(species_df, name) in {"new", "unusual", "tropical", "invasive"}


def save_species_list(df, path):
    """Write a site list back to CSV (expected as yes/no), after basic checks."""
    out = df.copy()
    for col in ["common_name", "scientific_name", "status"]:
        out[col] = out[col].fillna("").astype(str).str.strip()
    out = out[(out["common_name"] != "") & (out["scientific_name"] != "")]
    out["status"] = out["status"].str.lower()
    bad = set(out["status"]) - STATUSES
    if bad:
        raise ValueError(f"Unknown status: {', '.join(sorted(bad))}")
    out["expected"] = ["yes" if bool(v) else "no" for v in out["expected"].fillna(False)]
    for col in EXTRA_COLUMNS:
        out[col] = out[col].fillna("").astype(str).str.strip() if col in out else ""
    out[BASE_COLUMNS + EXTRA_COLUMNS].to_csv(path, index=False)
    return len(out)


def load_species_list(path):
    """Read a site CSV and clean it up. Raises a clear error if a column is missing."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"common_name", "scientific_name", "status", "expected"} - set(df.columns)
    if missing:
        raise ValueError(f"{Path(path).name} is missing column(s): {', '.join(sorted(missing))}")
    for col in ["common_name", "scientific_name", "status"]:
        df[col] = df[col].astype(str).str.strip()
    df["status"] = df["status"].str.lower()
    df["expected"] = df["expected"].astype(str).str.strip().str.lower().isin(["yes", "y", "true", "1"])
    bad = set(df["status"]) - STATUSES
    if bad:
        raise ValueError(f"Unknown status value(s) in {Path(path).name}: {', '.join(sorted(bad))}")
    for col in EXTRA_COLUMNS:  # older lists don't have these yet
        df[col] = df[col].fillna("").astype(str).str.strip() if col in df else ""
    return df


def add_to_list(path, name, status, first_seen="", evidence="", common=""):
    """Add a species a person just confirmed to a site's list (a new arrival).

    It goes in as "not usually found here", with the date and where the proof is.
    If it's already on the list, only the empty first_seen / evidence fields are filled in.
    """
    df = load_species_list(path)
    match = df["scientific_name"].str.lower() == name.strip().lower()
    if match.any():
        i = df.index[match][0]
        if not df.at[i, "first_seen"]:
            df.at[i, "first_seen"] = first_seen
        if not df.at[i, "evidence"]:
            df.at[i, "evidence"] = evidence
    else:
        row = {col: "" for col in EXTRA_COLUMNS}
        row.update(common_name=common.strip() or name.strip(), scientific_name=name.strip(),
                   status=status, expected=False, first_seen=first_seen, evidence=evidence)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_species_list(df, path)
    return path


def site_csv(site_name):
    """Find the CSV for a site name (the file name without .csv)."""
    path = config.SPECIES_DIR / f"{site_name}.csv"
    return path if path.exists() else None


def common_name(species_df, name):
    """Scientific name -> common name, if the species is on the list."""
    if species_df is None or not name:
        return name
    row = species_df[species_df["scientific_name"] == name]
    return row["common_name"].iloc[0] if len(row) else name


def decide_flag(top_species, top_score, visibility_score, species_df):
    """Put one AI suggestion into a review group.

    The order matters:
    1. Murky water first: a low score in murky water means "can't see", not "new species".
    2. Plants and seaweed caught by the detector are probably false alarms.
    3. Low confidence = unsure.
    4. Not expected at this site, or on the watch list = unexpected.
    """
    if visibility_score < config.MURKY_SCORE:
        return "unidentifiable"
    row = species_df[species_df["scientific_name"] == top_species]
    if row.empty:
        return "unsure"
    row = row.iloc[0]
    if row["status"] == "not_fish":
        return "false_alarm"
    if top_score < config.SPECIES_MIN_SCORE:
        return "unsure"
    if row["status"] in WATCH_STATUSES or not row["expected"]:
        return "unexpected"
    return "known"
