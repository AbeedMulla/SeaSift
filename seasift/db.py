"""The shared data format. Every module reads and writes these tables.

If you change a table, update README.md and tell the team first.
"""
import datetime
import sqlite3
from pathlib import Path

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id           INTEGER PRIMARY KEY,
    filename     TEXT,
    path         TEXT,     -- full path to the video file
    site         TEXT,     -- name of the species list used (file name without .csv)
    duration_s   REAL,
    fps          REAL,
    processed_at TEXT,
    ai_used      INTEGER DEFAULT 0,
    energy_kwh   REAL,     -- estimated electricity used to process this video
    survey_date  TEXT      -- the day the footage was recorded (YYYY-MM-DD)
);

-- Visibility, one row per window of VIS_WINDOW_S seconds
CREATE TABLE IF NOT EXISTS segments (
    id       INTEGER PRIMARY KEY,
    video_id INTEGER,
    start_s  REAL,
    end_s    REAL,
    score    REAL,         -- 0 (can't see) to 100 (crystal clear)
    label    TEXT          -- clear / murky / unusable / blocked
);

-- Moments where something moved
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY,
    video_id   INTEGER,
    start_s    REAL,
    end_s      REAL,
    peak_s     REAL,       -- the moment with the most movement
    motion     REAL,       -- share of the frame that moved at the peak
    visibility REAL,       -- median visibility score during the event
    thumb_path TEXT,       -- picture of the peak moment (relative to the project folder)
    status     TEXT DEFAULT 'pending',  -- pending / reviewed / false_alarm (nothing there) / camera (handling)
    boxes      TEXT        -- where it moved at the peak: JSON list of [x, y, w, h], shares of the frame (0-1)
);

-- AI tier only: one row per fish box the detector found
CREATE TABLE IF NOT EXISTS detections (
    id        INTEGER PRIMARY KEY,
    event_id  INTEGER,
    video_id  INTEGER,
    frame_s   REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    det_conf  REAL,
    crop_path TEXT,
    species_1 TEXT, score_1 REAL,
    species_2 TEXT, score_2 REAL,
    species_3 TEXT, score_3 REAL,
    flag      TEXT         -- known / unexpected / unidentifiable / unsure / false_alarm
);

-- What a person confirmed. count = the most of that species seen at the same time.
CREATE TABLE IF NOT EXISTS labels (
    id         INTEGER PRIMARY KEY,
    event_id   INTEGER,
    video_id   INTEGER,
    species    TEXT,       -- scientific name if it's on the list, otherwise what was typed
    count      INTEGER,
    note       TEXT,
    labeled_at TEXT,
    proof_s    REAL,       -- the moment (seconds) the reviewer picked as proof of the count
    traits     TEXT,       -- what the reviewer could see, e.g. "body: torpedo, tail: forked"
    photo_path TEXT        -- small picture of the proof moment, used on the species cards
);

-- Protocol mode (MeanCount): the equally spaced frames counted in each video
CREATE TABLE IF NOT EXISTS subsamples (
    id         INTEGER PRIMARY KEY,
    video_id   INTEGER,
    n          INTEGER,    -- sub-sample number, 1, 2, 3...
    time_s     REAL,
    visibility TEXT,       -- <0.5 m / 0.5-1.0 m / 1.0-1.5 m / >1.5 m
    notes      TEXT,
    counter    TEXT,       -- initials of who counted it
    done       INTEGER DEFAULT 0,
    qc         INTEGER DEFAULT 0,  -- 1 = picked for the blind second count
    qc_counter TEXT
);

-- Protocol mode: one dot per animal counted in a sub-sample frame
CREATE TABLE IF NOT EXISTS dots (
    id           INTEGER PRIMARY KEY,
    subsample_id INTEGER,
    pass         INTEGER DEFAULT 1,  -- 1 = the count, 2 = the blind quality-check count
    x REAL, y REAL,                  -- where, as shares of the frame (0-1)
    kind         TEXT,               -- fish / unknown (fish, species unknown) / invert
    species      TEXT,
    counter      TEXT,
    added_at     TEXT
);
"""


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def connect(path=None):
    """Open the database (and create the tables the first time)."""
    path = Path(path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.executescript(SCHEMA)
    # Databases made by earlier versions miss some columns: add them
    add_missing_column(conn, "labels", "proof_s", "REAL")
    add_missing_column(conn, "events", "boxes", "TEXT")
    add_missing_column(conn, "videos", "survey_date", "TEXT")
    add_missing_column(conn, "labels", "traits", "TEXT")
    add_missing_column(conn, "labels", "photo_path", "TEXT")
    add_missing_column(conn, "videos", "juveniles", "INTEGER")
    return conn


def add_missing_column(conn, table, column, kind):
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
        conn.commit()


def query(conn, sql, params=()):
    """Run a SELECT and get the result as a pandas table."""
    return pd.read_sql_query(sql, conn, params=params)


def add_video(conn, path, site, duration_s, fps, ai_used, survey_date=None):
    path = Path(path)
    cur = conn.execute(
        "INSERT INTO videos (filename, path, site, duration_s, fps, processed_at, ai_used, survey_date) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (path.name, str(path.resolve()), site, duration_s, fps, now(), int(ai_used), survey_date),
    )
    conn.commit()
    return cur.lastrowid


def set_energy(conn, video_id, kwh):
    conn.execute("UPDATE videos SET energy_kwh = ? WHERE id = ?", (kwh, video_id))
    conn.commit()


def add_segments(conn, video_id, segments):
    conn.executemany(
        "INSERT INTO segments (video_id, start_s, end_s, score, label) VALUES (?, ?, ?, ?, ?)",
        [(video_id, s["start_s"], s["end_s"], s["score"], s["label"]) for s in segments],
    )
    conn.commit()


def add_events(conn, video_id, events):
    """Save events and write the new database id back into each event dict."""
    for ev in events:
        cur = conn.execute(
            "INSERT INTO events (video_id, start_s, end_s, peak_s, motion, visibility, thumb_path, boxes, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (video_id, ev["start_s"], ev["end_s"], ev["peak_s"], ev["motion"],
             ev["visibility"], ev.get("thumb_path"), ev.get("boxes"), ev.get("status", "pending")),
        )
        ev["id"] = cur.lastrowid
    conn.commit()


def add_detections(conn, rows):
    cols = ["event_id", "video_id", "frame_s", "x1", "y1", "x2", "y2", "det_conf", "crop_path",
            "species_1", "score_1", "species_2", "score_2", "species_3", "score_3", "flag"]
    conn.executemany(
        f"INSERT INTO detections ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        [tuple(r[c] for c in cols) for r in rows],
    )
    conn.commit()


def save_label(conn, event_id, video_id, species, count, note="", proof_s=None, traits=None):
    """Save what a person confirmed. Returns the new label's id."""
    cur = conn.execute(
        "INSERT INTO labels (event_id, video_id, species, count, note, labeled_at, proof_s, traits) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (event_id, video_id, species, int(count), note, now(), proof_s, traits),
    )
    conn.commit()
    return cur.lastrowid


def set_label_photo(conn, label_id, photo_path):
    conn.execute("UPDATE labels SET photo_path = ? WHERE id = ?", (photo_path, label_id))
    conn.commit()


def delete_label(conn, label_id):
    conn.execute("DELETE FROM labels WHERE id = ?", (label_id,))
    conn.commit()


def set_event_status(conn, event_id, status):
    conn.execute("UPDATE events SET status = ? WHERE id = ?", (status, event_id))
    conn.commit()


# ---------------------------------------------------------------- protocol mode (MeanCount)
def create_subsamples(conn, video_id, times):
    conn.executemany("INSERT INTO subsamples (video_id, n, time_s) VALUES (?, ?, ?)",
                     [(video_id, i + 1, t) for i, t in enumerate(times)])
    conn.commit()


def delete_subsamples(conn, video_id):
    """Start the count over for one video (removes its frames and dots)."""
    conn.execute("DELETE FROM dots WHERE subsample_id IN (SELECT id FROM subsamples WHERE video_id = ?)",
                 (video_id,))
    conn.execute("DELETE FROM subsamples WHERE video_id = ?", (video_id,))
    conn.commit()


def update_subsample(conn, subsample_id, **fields):
    """e.g. update_subsample(conn, 3, visibility="0.5-1.0 m", done=1)"""
    for name, value in fields.items():
        assert name in {"visibility", "notes", "counter", "done", "qc", "qc_counter"}, name
        conn.execute(f"UPDATE subsamples SET {name} = ? WHERE id = ?", (value, subsample_id))
    conn.commit()


def add_dot(conn, subsample_id, x, y, kind, species="", counter="", pass_no=1):
    conn.execute("INSERT INTO dots (subsample_id, pass, x, y, kind, species, counter, added_at) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (subsample_id, pass_no, x, y, kind, species, counter, now()))
    conn.commit()


def delete_dot(conn, dot_id):
    conn.execute("DELETE FROM dots WHERE id = ?", (dot_id,))
    conn.commit()


def set_juveniles(conn, video_id, present):
    conn.execute("UPDATE videos SET juveniles = ? WHERE id = ?", (present, video_id))
    conn.commit()
