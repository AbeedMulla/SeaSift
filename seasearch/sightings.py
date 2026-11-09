"""Early warning for new arrivals: which species were confirmed at each site, and when.

Built only from what people confirmed on the Review page (the labels table). No AI.
  - first sighting: a species confirmed at a site for the first time
  - new arrivals:   confirmed species that aren't usual there (not on the list, not usually found
                    there, tropical strays, invasive) plus species at risk
  - not seen lately: usual species whose last sighting is old (NOT_SEEN_DAYS)
"""
import datetime

import pandas as pd

from . import config, db, species

ALERT_CATEGORIES = {"new", "unusual", "tropical", "invasive", "at_risk"}
COLUMNS = ["species", "first_seen", "last_seen", "surveys", "sightings"]


def survey_day(video):
    """The survey date of a video row (YYYY-MM-DD). Older rows fall back to the processing date."""
    day = video.get("survey_date") if hasattr(video, "get") else None
    return day if isinstance(day, str) and day else str(video["processed_at"])[:10]


def site_sightings(conn, site):
    """Every confirmed sighting at a site: species, survey date, video, proof time, picture."""
    return db.query(
        conn,
        "SELECT l.id AS label_id, l.species, l.count, l.photo_path, l.video_id, v.filename, "
        "COALESCE(v.survey_date, substr(v.processed_at, 1, 10)) AS survey_date, "
        "COALESCE(l.proof_s, e.peak_s) AS time_s "
        "FROM labels l JOIN videos v ON l.video_id = v.id JOIN events e ON l.event_id = e.id "
        "WHERE v.site = ? ORDER BY survey_date, l.id", (site,))


def is_first_sighting(conn, site, name, label_id):
    """True if no other confirmed sighting of this species exists at this site."""
    others = db.query(conn, "SELECT COUNT(*) AS n FROM labels l JOIN videos v ON l.video_id = v.id "
                            "WHERE v.site = ? AND l.species = ? AND l.id != ?", (site, name, label_id))
    return int(others.n.iloc[0]) == 0


def needs_alert(species_df, name):
    """Alert for species that aren't usual at this site, and for species at risk."""
    return species.species_category(species_df, name) in ALERT_CATEGORIES


def summary(conn, site, species_df):
    """One row per species confirmed at a site: first and last seen, surveys, sightings, tag."""
    rows = site_sightings(conn, site)
    if rows.empty:
        return pd.DataFrame(columns=COLUMNS + ["common_name", "category"])
    table = (rows.groupby("species")
             .agg(first_seen=("survey_date", "min"), last_seen=("survey_date", "max"),
                  surveys=("video_id", "nunique"), sightings=("label_id", "count"))
             .reset_index())
    table["common_name"] = [species.common_name(species_df, n) for n in table.species]
    table["category"] = [species.species_category(species_df, n) for n in table.species]
    return table.sort_values("first_seen", ascending=False).reset_index(drop=True)


def new_arrivals(table):
    """The species to tell Coastal Action about, newest first."""
    return table[table.category.isin(ALERT_CATEGORIES)].reset_index(drop=True)


def not_seen_lately(table, species_df, days=config.NOT_SEEN_DAYS):
    """Usual species seen before, but not in the `days` before the latest survey at this site.

    Returns (not seen lately table, list of usual species never confirmed yet)."""
    if species_df is None or table.empty:
        return table.iloc[0:0], []
    usual = species_df[(species_df.status != "not_fish") & species_df.expected]
    latest = datetime.date.fromisoformat(table.last_seen.max())
    cutoff = (latest - datetime.timedelta(days=days)).isoformat()
    seen = table[table.species.isin(usual.scientific_name)]
    old = seen[seen.last_seen < cutoff].reset_index(drop=True)
    never = [r.common_name for r in usual.itertuples() if r.scientific_name not in set(table.species)]
    return old, never


def first_seen_in_video(conn, site, video_id):
    """Species whose first sighting at this site happened in this video."""
    rows = site_sightings(conn, site)
    if rows.empty:
        return []
    first = rows.sort_values(["survey_date", "label_id"]).groupby("species").head(1)
    return list(first[first.video_id == video_id].species)
