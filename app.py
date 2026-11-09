"""seaSEArch app. Start it with:   streamlit run app.py

Pages:
  1 · Process a video   -> triage + footage quality (+ optional local AI); saves every review picture
  2 · Review moments    -> one moment at a time: confirm species and count (camera handling skipped)
  3 · Report            -> MaxN per species with proof frames, usable minutes, CSV downloads
  4 · Sightings         -> new arrivals and watch-list species per site, species not seen lately
  5 · Count frames      -> Coastal Action's MeanCount protocol: equally spaced frames, click each fish
  Species lists         -> edit each site's list (status, group, ID features for the species cards)

Smooth on 4K video: pages never decode video while you click. Pictures are saved once (see
review_picture and count_frame), and clips are made one at a time in the background (ClipMaker).
"""
import datetime
import threading
import time
from pathlib import Path

import altair as alt
import cv2
import pandas as pd
import streamlit as st

from streamlit_image_coordinates import streamlit_image_coordinates

from seasearch import cards, clips, config, counting, db, difficulty, protocol, sightings, species
from seasearch.ai import ai_status
from seasearch.energy import describe_kwh
from seasearch.enhance import enhance
from seasearch.pipeline import (mark_camera_handling, mark_scenery, process_video, recorded_date, review_moments,
                              thumb_folder, unprocessed_videos)
from seasearch.report import build_report_html
from seasearch.video import FrameReader, cached_frame
from seasearch.visibility import minutes_by_label

st.set_page_config(page_title="seaSEArch", page_icon="🐟", layout="wide")
# Less empty space above each page, so more of the work fits on one screen
st.markdown("<style>[data-testid='stMainBlockContainer'] {padding-top: 2.5rem;}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_conn():
    return db.connect()


# ---------------------------------------------------------------- pictures (decoded once, then instant)
@st.cache_data(show_spinner=False, max_entries=120)  # about 1.5 MB each
def review_picture(video_id, path, t, enhanced=False):
    """A Review page picture. Read from the video once (processing usually does this already),
    then from disk, and kept in memory: moving between moments never waits for the video."""
    frame = cached_frame(path, t, thumb_folder(video_id), width=config.REVIEW_WIDTH)
    return enhance(frame) if frame is not None and enhanced else frame


@st.cache_data(show_spinner=False, max_entries=16)  # about 4 MB each
def count_frame(video_id, path, t, enhanced):
    """A big, sharp frame to count on. Saved once, so each click only redraws the dots."""
    frame = cached_frame(path, t, config.COUNT_FRAME_DIR / f"video_{video_id}", width=config.COUNT_WIDTH,
                         quality=95)
    return enhance(frame) if frame is not None and enhanced else frame


def save_count_frames(video_id, path, times, bar):
    """Save every count frame up front (one pass), so counting never waits for the video."""
    folder = config.COUNT_FRAME_DIR / f"video_{video_id}"
    with FrameReader(path) as reader:
        for i, t in enumerate(times):
            cached_frame(path, t, folder, width=config.COUNT_WIDTH, quality=95, reader=reader)
            bar.progress((i + 1) / len(times), text=f"Saving frame {i + 1} of {len(times)}...")


@st.cache_data(show_spinner=False)
def full_width(path):
    """The video's real width in pixels (the AI's boxes are measured on full-size frames)."""
    cap = cv2.VideoCapture(path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
    cap.release()
    return width


@st.cache_data(show_spinner=False)
def load_site(site_name):
    path = species.site_csv(site_name) if site_name else None
    return species.load_species_list(path) if path else None


conn = get_conn()


def fmt_time(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def show_image(path_or_frame, enhanced=False, **kwargs):
    frame = cv2.imread(str(config.full_path(path_or_frame))) if isinstance(path_or_frame, str) else path_or_frame
    if frame is None:
        st.caption("(picture not available)")
        return
    st.image(enhance(frame) if enhanced else frame, channels="BGR", **kwargs)


def draw_boxes(frame, boxes, scale=1.0):
    """Draw the AI's fish boxes (measured on the full-size frame) on a copy of a smaller frame."""
    out = frame.copy()
    thickness = max(2, frame.shape[1] // 400)
    for b in boxes.itertuples():
        cv2.rectangle(out, (int(b.x1 * scale), int(b.y1 * scale)), (int(b.x2 * scale), int(b.y2 * scale)),
                      (0, 255, 255), thickness)
    return out


# ---------------------------------------------------------------- video clips (made in the background)
class ClipMaker:
    """Makes video clips one at a time in a background thread, newest request first.

    The clip you're likely to want next gets made while you look at the pictures, so pressing
    play is instant. Only one clip is made at a time (older requests are dropped), so the
    laptop stays cool and the app stays smooth.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.waiting = None      # (name, make): the newest request
        self.busy = None         # the name of the clip being made in the background right now
        self.making_now = set()  # clips someone is waiting for, made right away (not in the background)
        self.thread = None

    def request(self, name, make):
        with self.lock:
            if self.busy == name or name in self.making_now or clip_file(name):
                return
            self.waiting = (name, make)
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._work, daemon=True)
                self.thread.start()

    def _work(self):
        while True:
            with self.lock:
                if self.waiting is None:
                    self.busy = None
                    return
                (name, make), self.waiting = self.waiting, None
                if name in self.making_now:
                    continue
                self.busy = name
            if not clip_file(name):
                try:
                    make()
                except Exception:  # a broken clip just isn't shown; the pictures still work
                    pass

    def wait_for(self, name, timeout=300):
        """Wait while the background thread finishes this clip."""
        end = time.time() + timeout
        while self.busy == name and time.time() < end:
            time.sleep(0.2)

    def make_now(self, name, make):
        """Make a clip right away, because someone is waiting for it (the background skips it)."""
        with self.lock:
            self.making_now.add(name)
            if self.waiting and self.waiting[0] == name:
                self.waiting = None
        try:
            if clip_file(name) is None:
                make()
        finally:
            with self.lock:
                self.making_now.discard(name)


@st.cache_resource
def clip_maker():
    return ClipMaker()


def clip_file(name):
    """The finished clip with this name (.mp4 or .webm, whichever this computer can write), or None."""
    for ext in (".mp4", ".webm"):
        path = Path(f"{name}{ext}")
        if path.exists():
            return path
    return None


def make_clip_file(video_path, start_s, end_s, name, **kwargs):
    """Make a clip under a temporary name, then rename it, so a half-made clip is never shown."""
    made, _ = clips.make_clip(video_path, start_s, end_s, f"{name}_making", **kwargs)
    if made is None:
        return None
    final = Path(f"{name}{made.suffix}")
    made.replace(final)
    return final


def event_clip_job(video, ev, enhanced):
    """(name, how to make it) for one event's clip, enhanced or in the original colours."""
    end_s = min(float(ev.end_s), float(ev.start_s) + config.MAX_CLIP_S)
    name = str(config.CLIP_DIR / f"video{int(video.id)}_{'enh' if enhanced else 'orig'}_"
                                 f"{round(ev.start_s * 10)}-{round(end_s * 10)}")
    return name, lambda: make_clip_file(str(video.path), float(ev.start_s), end_s, name, enhanced=enhanced)


def context_clip_job(video, t):
    """(name, how to make it) for a count frame's clip: 5 s before and after, at quarter speed."""
    name = str(config.CLIP_DIR / f"video{int(video.id)}_ctx_{round(t * 10)}")
    return name, lambda: make_clip_file(str(video.path), max(0.0, t - config.CONTEXT_S), t + config.CONTEXT_S,
                                        name, width=960, fps_out=30, speed=config.CONTEXT_SPEED)


def get_clip(job, wait_text):
    """A finished clip: right away if it's made, else wait for the background thread, else make it now."""
    name, make = job
    maker = clip_maker()
    if clip_file(name) is None:
        with st.spinner(wait_text):
            if maker.busy == name:
                maker.wait_for(name)
            if clip_file(name) is None:
                maker.make_now(name, make)
    return clip_file(name)


def play_clip(path):
    st.video(str(path), format="video/webm" if path.suffix == ".webm" else "video/mp4",
             loop=True, autoplay=True, muted=True)


# ---------------------------------------------------------------- water clarity chart
QUALITY_COLORS = {"clear": "#0ca30c", "murky": "#fab219", "unusable": "#ec835a", "blocked": "#d03b3b"}


def minute_quality(segments):
    """Footage quality minute by minute: the median clarity score of each minute's 10-second windows."""
    rows = []
    if segments.empty:
        return pd.DataFrame(columns=["start", "end", "score", "label", "minute"])
    for minute, g in segments.assign(minute=(segments.start_s // 60).astype(int)).groupby("minute"):
        score = float(g.score.median())
        label = ("blocked" if (g.label == "blocked").mean() > 0.5 else "clear" if score >= config.CLEAR_SCORE
                 else "murky" if score >= config.MURKY_SCORE else "unusable")
        rows.append({"start": minute + 0.06, "end": minute + 0.94, "score": round(score), "label": label,
                     "minute": f"{minute}:00 to {minute + 1}:00"})
    return pd.DataFrame(rows)


def clarity_chart(segments, frame_times=(), current=None, height=230):
    """Water clarity minute by minute (bar height and colour), with the count frames marked."""
    data = minute_quality(segments)
    end = float(data.end.max() + 0.06) if not data.empty else 1.0
    x_scale = alt.Scale(domain=[0, end], nice=False)
    bars = alt.Chart(data).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("start:Q", scale=x_scale, title="Minute of the video",
                axis=alt.Axis(grid=False, tickMinStep=1, labelColor="#5f5e5a", titleColor="#5f5e5a",
                              domainColor="#c3c2b7", tickColor="#c3c2b7")),
        x2="end:Q",
        y2=alt.datum(0),  # bars grow up from zero
        y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 115]), title="Clarity",
                axis=alt.Axis(values=[0, config.MURKY_SCORE, config.CLEAR_SCORE, 100], gridColor="#e1e0d9",
                              labelColor="#5f5e5a", titleColor="#5f5e5a", domain=False, ticks=False)),
        color=alt.Color("label:N", title=None,
                        scale=alt.Scale(domain=list(QUALITY_COLORS), range=list(QUALITY_COLORS.values())),
                        legend=alt.Legend(orient="top", direction="horizontal", labelColor="#3d3d3a")),
        tooltip=[alt.Tooltip("minute:N", title="Minute"), alt.Tooltip("label:N", title="Footage"),
                 alt.Tooltip("score:Q", title="Clarity score (0-100)")])
    layers = [bars]
    if len(frame_times):
        frames = pd.DataFrame({"t": [t / 60 for t in frame_times], "frame": range(1, len(frame_times) + 1),
                               "time": [fmt_time(t) for t in frame_times]})
        window = pd.DataFrame({"a": [frames.t.min() - 0.3], "b": [frames.t.max() + 0.3]})
        layers.insert(0, alt.Chart(window).mark_rect(color="#0b0b0b", opacity=0.06).encode(
            x=alt.X("a:Q", scale=x_scale), x2="b:Q"))
        layers.append(alt.Chart(frames).mark_tick(color="#3d3d3a", thickness=2, size=12).encode(
            x=alt.X("t:Q", scale=x_scale), y=alt.value(8),
            tooltip=[alt.Tooltip("frame:Q", title="Count frame"), alt.Tooltip("time:N", title="At")]))
    if current is not None:
        layers.append(alt.Chart(pd.DataFrame({"t": [current / 60]})).mark_rule(color="#0b0b0b", strokeWidth=2)
                      .encode(x=alt.X("t:Q", scale=x_scale)))
    return alt.layer(*layers).properties(height=height).configure_view(stroke=None)


PAGES = ["1 · Process a video", "2 · Review moments", "3 · Report", "4 · Sightings",
         "5 · Count frames (MeanCount)", "Species lists"]


def go_to(page_name, video_id=None):
    """For the 'next step' buttons (as on_click, so it runs before the sidebar is drawn)."""
    st.session_state["page"] = page_name
    if video_id is not None:
        st.session_state["current_video"] = int(video_id)


def pick_video(key):
    """Pick a processed video. Every page opens on the video you used last, so you only pick it once."""
    videos = db.query(conn, "SELECT * FROM videos ORDER BY id DESC")
    if videos.empty:
        st.info("No processed videos yet. Start on page 1.")
        st.stop()
    names = {int(r.id): f"#{r.id} · {r.filename} · {r.site or 'no site list'}" for r in videos.itertuples()}
    ids = list(names)
    current = st.session_state.get("current_video")
    vid = st.selectbox("Video", ids, index=ids.index(current) if current in ids else 0,
                       format_func=names.get, key=key)
    st.session_state["current_video"] = vid
    return videos[videos.id == vid].iloc[0]


def save_species(ev, video, site_df, name, count, proof, note="", finish=False):
    """Save one species for a moment, with what the reviewer saw and a proof picture.
    finish: the moment is done too (the reviewer moves on to the next one)."""
    vid, event_id = int(video.id), int(ev.id)
    seen_now = {f: st.session_state.get(f"q_{f}_{event_id}") for f in ["what", *cards.QUESTIONS]}
    label_id = db.save_label(conn, event_id, vid, name, count, note, proof_s=proof,
                             traits=cards.describe(seen_now) or None)
    frame = review_picture(vid, str(video.path), proof, bool(ev.visibility < config.CLEAR_SCORE))
    if frame is not None:  # a small picture for the species cards
        db.set_label_photo(conn, label_id, cards.save_label_photo(frame, label_id))
    if (video.site and sightings.needs_alert(site_df, name)
            and sightings.is_first_sighting(conn, video.site, name, label_id)):
        st.session_state["new_arrival"] = {
            "label_id": label_id, "species": name, "site": video.site, "video_id": vid,
            "filename": video.filename, "time_s": float(proof), "survey_date": sightings.survey_day(video)}
    if finish:
        db.set_event_status(conn, event_id, "reviewed")
    common = species.common_name(site_df, name)
    st.session_state["last_saved"] = {"label_id": label_id, "event_id": event_id, "video_id": vid,
                                      "finished": finish, "text": f"{common} x{count} at {fmt_time(ev.start_s)}"}
    st.toast(f"Saved {common} x{count}" + (". Next moment" if finish else ""))


def undo_last_save():
    """Take back the last save, and go back to that moment if it moved you on."""
    last = st.session_state.pop("last_saved", None)
    if not last:
        return
    db.delete_label(conn, last["label_id"])
    if last["finished"]:
        db.set_event_status(conn, last["event_id"], "pending")
    st.session_state["jump_to_event"] = last["event_id"]
    if (st.session_state.get("new_arrival") or {}).get("label_id") == last["label_id"]:
        del st.session_state["new_arrival"]


def undo_line(video_id):
    """'Saved Cunner x2 at 0:42 [Undo]', after the last save."""
    last = st.session_state.get("last_saved")
    if not last or last["video_id"] != video_id:
        return
    row = st.columns([4, 1], vertical_alignment="center")
    row[0].markdown(f":green[:material/check_circle: Saved {last['text']}]")
    row[1].button("Undo", key="undo_save", on_click=undo_last_save, icon=":material/undo:", width="stretch")


def video_files():
    config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in config.VIDEO_DIR.iterdir() if p.suffix.lower() in config.VIDEO_TYPES)


def upload_box():
    up = st.file_uploader("Add a video", type=[t.strip(".") for t in config.VIDEO_TYPES])
    if up is not None:
        dest = config.VIDEO_DIR / up.name
        if not dest.exists():
            dest.write_bytes(up.getbuffer())
        st.success(f"Saved {up.name} to data/videos")


@st.cache_data(show_spinner=False, max_entries=200)
def card_picture(path):
    """A species photo cut to the same shape as the drawn outlines (16:9), so the cards line up."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if w / h > 16 / 9:
        cut = int((w - h * 16 / 9) / 2)
        img = img[:, cut:w - cut]
    else:
        cut = int((h - w * 9 / 16) / 2)
        img = img[cut:h - cut]
    return cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)


def species_cards(names, rows, seen, ai_scores, photos, event_id, saved_names, on_pick, first_number=1):
    """Species cards, three per row. One click on a card saves that species (or press its number)."""
    for start in range(0, len(names), 3):
        cols = st.columns(3)
        for i, (col, name) in enumerate(zip(cols, names[start:start + 3])):
            number, row = first_number + start + i, rows[name]
            with col.container(border=True):
                path, credit = cards.card_photo(row, photos)
                picture = card_picture(path) if path is not None else None
                if picture is None:
                    picture, credit = cards.outline(row.get("body", ""), species.group_of(row)), None
                own = name in photos  # a picture from your own footage needs no credit line
                st.image(picture, channels="BGR", width="stretch", caption=None if own else credit or None)
                st.markdown(f"**{row['common_name']}** {species.badge(species.row_category(row))}")
                facts = [f"*{name}*", f"seen here {seen.get(name, 0)}x" + (", your photo" if own else "")]
                if name in ai_scores:
                    facts.append(f"AI match {ai_scores[name]:.0%}")
                st.caption(" · ".join(facts))
                looks = cards.describe_features(row)
                tip = "  \n".join(x for x in [row.get("tip") or "", f"Looks: {looks}" if looks else ""] if x)
                saved = name in saved_names
                has_key = number <= 9 and not saved
                if st.button("Saved" if saved else "This one",  # Streamlit shows the number key on it
                             key=f"card_{event_id}_{name}", width="stretch", disabled=saved,
                             icon=":material/check:" if saved else None, help=tip or None,
                             shortcut=str(number) if has_key else None):
                    on_pick(name)
                    st.rerun()  # the whole page: the next moment, or this one with the species saved


def show_new_arrival(arrival, site_df):
    """The first-sighting alert, shown after a reviewer confirms a species new to this site."""
    name = arrival["species"]
    category = species.species_category(site_df, name)
    with st.container(border=True):
        cols = st.columns([1, 2])
        photo = db.query(conn, "SELECT photo_path FROM labels WHERE id = ?", (arrival["label_id"],))
        path = photo.photo_path.iloc[0] if not photo.empty else None
        if isinstance(path, str) and config.full_path(path).exists():
            cols[0].image(str(config.full_path(path)), width="stretch", caption="Proof picture")
        with cols[1]:
            st.markdown(f"### 🆕 First sighting at this site: {species.common_name(site_df, name)}")
            st.markdown(species.badge(category))
            st.write(f"Survey {arrival['survey_date']} · {arrival['filename']} at {fmt_time(arrival['time_s'])}"
                     f" · site: {arrival['site']}")
            evidence = f"{arrival['filename']} at {fmt_time(arrival['time_s'])}"
            path = species.site_csv(arrival["site"])
            if category == "new":
                st.write("It's not on this site's species list. Add it, with this sighting as the evidence?")
                status_pick = st.selectbox("Status", ["native", "invasive", "tropical_stray", "at_risk", "not_fish"],
                                           key="arrival_status")
                common = st.text_input("Common name (optional)", key="arrival_common")
                b = st.columns(2)
                if b[0].button("Add to species list", type="primary", disabled=path is None):
                    species.add_to_list(path, name, status_pick, arrival["survey_date"], evidence, common)
                    load_site.clear()
                    del st.session_state["new_arrival"]
                    st.toast(f"Added {name} to {arrival['site']}")
                    st.rerun()
                if b[1].button("Not now"):
                    del st.session_state["new_arrival"]
                    st.rerun()
            else:
                st.write({"invasive": "Invasive species: let the team know quickly.",
                          "tropical": "Tropical stray: a sign of warming water, worth reporting.",
                          "at_risk": "Species at risk: share the exact location carefully.",
                          "unusual": "Not usually found here: worth a second look by an expert."}[category])
                if st.button("Noted", type="primary"):
                    if path is not None:  # fills in first seen + evidence on the list if they're empty
                        species.add_to_list(path, name, "native", arrival["survey_date"], evidence)
                        load_site.clear()
                    del st.session_state["new_arrival"]
                    st.rerun()


@st.fragment
def help_me_identify(event_id, site, site_df, dets, saved_names, on_pick):
    """Quick questions narrow down the species cards (no AI); one click on a card saves it.
    Answering a question only redraws this part of the page."""
    what = st.pills("What is it?", list(cards.WHAT), format_func=cards.WHAT.get, key=f"q_what_{event_id}",
                    help="Answer only what you can see: each answer hides the species that don't match. "
                         "Can't tell? Leave it blank.")
    answers = {"what": what}
    if what != "invertebrate":  # the shape questions are about fish
        answered = any(st.session_state.get(f"q_{f}_{event_id}") for f in cards.QUESTIONS)
        with st.expander("Not sure which fish? Narrow it down by shape", expanded=answered):
            answers |= {feature: st.pills(feature.capitalize(), choices, key=f"q_{feature}_{event_id}")
                        for feature, choices in cards.QUESTIONS.items()}
    listed = site_df[site_df.status != "not_fish"]
    shown = cards.filter_species(listed, answers)
    ai_scores = cards.ai_scores_for_event(dets) if not dets.empty else {}
    seen = cards.seen_counts_at_site(conn, site)
    photos = cards.own_photos(conn)
    order = cards.card_order(shown, seen, ai_scores)
    rows = {r["scientific_name"]: r for r in shown.to_dict("records")}
    st.caption(f"{len(order)} of {len(listed)} species on this site's list match · click a card or press "
               "its number to save · :orange[sample data]",
               help="The species descriptions and tips are our drafts for this prototype, not checked by "
                    "a biologist. Edit them on the Species lists page. Hover over a card's button for its tip.")
    species_cards(order[:6], rows, seen, ai_scores, photos, event_id, saved_names, on_pick)
    if len(order) > 6 and st.toggle(f"Show {len(order) - 6} more", key=f"more_cards_{event_id}"):
        species_cards(order[6:], rows, seen, ai_scores, photos, event_id, saved_names, on_pick,
                      first_number=7)  # only drawn when asked


@st.fragment
def counting_area(video, subs, site_df, qc_mode):
    """Count one frame at a time. Clicking (a dot, a frame, Undo) only redraws this part of the page."""
    vid, path = int(video.id), str(video.path)
    pass_no = 2 if qc_mode else 1
    view = subs[subs.qc == 1] if qc_mode else subs
    pos_key = f"count_pos_{vid}_{pass_no}"
    pos = min(st.session_state.get(pos_key, 0), len(view) - 1)
    finished = view.qc_counter.notna() if qc_mode else view.done == 1

    def go(new_pos):
        st.session_state[pos_key] = min(max(new_pos, 0), len(view) - 1)

    nav = st.columns([2, 2, 5], vertical_alignment="center")
    nav[0].button("Previous frame", on_click=go, args=(pos - 1,), disabled=pos == 0)
    nav[1].button("Next frame", on_click=go, args=(pos + 1,), disabled=pos >= len(view) - 1)
    goto_key = f"goto_{vid}_{pass_no}_{pos}"  # a new box per frame, so it always shows the current one
    nav[2].selectbox("Go to frame", range(len(view)), index=pos, key=goto_key, label_visibility="collapsed",
                     on_change=lambda: go(st.session_state[goto_key]),
                     format_func=lambda i: f"Frame {int(view.n.iloc[i])} · {fmt_time(view.time_s.iloc[i])}"
                                           + (" · counted" if finished.iloc[i] else ""))
    s = view.iloc[pos]
    dots = db.query(conn, "SELECT * FROM dots WHERE subsample_id = ? AND pass = ? ORDER BY id", (int(s.id), pass_no))
    records = dots.to_dict("records")
    counts = protocol.frame_counts(records)
    context_job = context_clip_job(video, float(s.time_s))
    clip_maker().request(*context_job)  # made in the background, ready by the time you want to watch it
    left, right = st.columns([3, 1])

    with right:
        st.subheader(f"Frame {int(s.n)} of {len(subs)}")
        st.write(f"at **{fmt_time(s.time_s)}**" + (" · second count, blind" if qc_mode else ""))
        nearby = db.query(conn, "SELECT COUNT(*) AS n FROM events WHERE video_id = ? AND status != 'camera' "
                                "AND start_s < ? AND end_s > ?",
                          (vid, float(s.time_s) + config.CONTEXT_S, float(s.time_s) - config.CONTEXT_S))
        st.markdown(":orange-badge[Movement seen within 5 s]" if int(nearby.n.iloc[0]) else
                    ":gray-badge[No movement seen within 5 s]",
                    help="From seaSEArch's movement scan. Either way, check the bottom for camouflaged animals.")
        kind = st.radio("What are you dotting?", list(protocol.KINDS), format_func=protocol.KINDS.get,
                        key="dot_kind")
        group = "invertebrate" if kind == "invert" else "fish"
        dot_species = "" if kind == "unknown" else st.selectbox(
            "Species", [""] + species.names_in_group(site_df, group), key=f"dot_species_{group}",
            format_func=lambda n: "Not named" if not n else f"{species.common_name(site_df, n)} ({n})")
        m = st.columns(2)
        m[0].metric("Fish", counts["fish"], border=True)
        m[1].metric("Invertebrates", counts["invertebrates"], border=True)
        for name, n in sorted(counts["species"].items()):
            st.write(f"{species.common_name(site_df, name)}: {n}")
        if counts["unknown"]:
            st.caption(f"{counts['unknown']} fish of unknown species")
        if not qc_mode:
            st.radio("Visibility", protocol.VISIBILITY, key=f"vis_{s.id}",
                     index=protocol.VISIBILITY.index(s.visibility) if s.visibility in protocol.VISIBILITY else None)
            st.text_area("Notes", value=protocol.text(s.notes), key=f"notes_{s.id}", height=80)

        def undo():
            db.delete_dot(conn, int(dots.id.iloc[-1]))

        def clear():
            for dot_id in dots.id:
                db.delete_dot(conn, int(dot_id))

        def save_frame():
            sub_id = int(s.id)
            if qc_mode:
                db.update_subsample(conn, sub_id, qc_counter=st.session_state.get("counter") or "second counter")
            else:
                db.update_subsample(conn, sub_id, visibility=st.session_state.get(f"vis_{sub_id}"),
                                    notes=st.session_state.get(f"notes_{sub_id}", ""),
                                    counter=st.session_state.get("counter", ""), done=1)
            go(pos + 1)
            st.rerun()  # the whole page, so MeanCount below is up to date

        b = st.columns(2)
        b[0].button("Undo last dot", disabled=dots.empty, on_click=undo)
        b[1].button("Clear frame", disabled=dots.empty, on_click=clear)
        st.button("Save this frame, go to the next", type="primary", shortcut="Mod+Enter", on_click=save_frame)
        st.caption("Only count an animal if you can see at least 2 of: its shape, body features (fins, eyes), "
                   "colour or flashes, or the way it moves. (Coastal Action protocol)")

    with left:
        opts = st.columns(2)
        enhanced = opts[0].toggle("Enhance", value=True, key="count_enhance")
        grid = opts[1].toggle("Grid with the fish count in each cell", value=True, key="count_grid")
        frame = count_frame(vid, path, float(s.time_s), enhanced)
        if frame is None:
            st.error("Couldn't read this frame from the video.")
            return
        img_key = f"img_{s.id}_{pass_no}"

        def add_or_remove_dot():
            click = st.session_state.get(img_key)
            if not click:
                return
            x, y = click["x"] / click["width"], click["y"] / click["height"]
            hit = protocol.nearest_dot(records, x, y, aspect=frame.shape[1] / frame.shape[0])
            if hit:
                db.delete_dot(conn, int(hit["id"]))
            else:
                db.add_dot(conn, int(s.id), x, y, kind, dot_species, st.session_state.get("counter", ""), pass_no)

        picture = protocol.draw(frame, records, grid=grid)
        streamlit_image_coordinates(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB), key=img_key, width="stretch",
                                    image_format="JPEG", jpeg_quality=90, cursor="crosshair",
                                    on_click=add_or_remove_dot)
        st.caption("Click each fish or invertebrate to put a dot on it. Click a dot again to remove it. "
                   "Yellow = fish, cyan = fish of unknown species, pink = invertebrate.")
        if st.toggle("Watch 5 s before and after, at quarter speed", key="count_context"):
            clip = get_clip(context_job, "Making the slow-motion clip (about half a minute for 4K, once)...")
            if clip:
                play_clip(clip)
            else:
                st.caption("This computer can't write video clips.")


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🐟 seaSEArch")
st.sidebar.caption("Hours of murky survey video, down to minutes of review. "
                   "Runs on this laptop: no internet, no trained AI model needed.")
page = st.sidebar.radio("Page", PAGES, key="page")
status = ai_status()
if status["ready"]:
    st.sidebar.caption("Local AI: ready (switch it on per video)")
else:
    st.sidebar.caption("Local AI: not set up. Everything else works without it.")
    with st.sidebar.expander("How to set up the AI tier"):
        for problem in status["problems"]:
            st.write("-", problem)


# ---------------------------------------------------------------- 1 · process
if page.startswith("1"):
    st.title("Process a survey video")
    st.caption("**1 · Find the movement**: skips empty footage and camera handling, and rates how clear the "
               "water is. → **2 · People confirm** each species. → **3 · Count and report**: MaxN, MeanCount "
               "and a one-file report.")
    upload_box()
    files = video_files()
    if not files:
        st.info("Upload a video above, or run `python tools/make_test_video.py` to make a test clip.")
        st.stop()
    video_path = st.selectbox("Video", files, format_func=lambda p: p.name)
    sites = sorted(config.SPECIES_DIR.glob("*.csv"))
    site = st.selectbox("Site species list", sites, format_func=lambda p: p.stem)
    survey_day = st.date_input("Survey date (the day this footage was recorded)",
                               value=datetime.date.fromisoformat(recorded_date(video_path)),
                               key=f"date_{video_path.name}")
    use_ai = st.toggle("Use local AI (fish detector + species suggestions)", value=False,
                       disabled=not status["ready"],
                       help="Runs on this laptop only. Suggestions are checked by a person on the next page.")

    waiting = unprocessed_videos(conn)
    if len(waiting) > 1:
        st.caption(f"{len(waiting)} videos in data/videos haven't been processed yet. "
                   "Processing them all uses each file's own date as the survey date.")
        if st.button(f"Process all {len(waiting)} new videos (one after another)"):
            bar = st.progress(0.0, text="Starting...")
            results = []
            for i, path in enumerate(waiting):
                def progress(f, m, i=i, name=path.name):
                    bar.progress(min(max((i + f) / len(waiting), 0.0), 1.0),
                                 text=f"Video {i + 1} of {len(waiting)} ({name}): {m}")
                try:
                    vid, s = process_video(path, site=site, use_ai=use_ai, conn=conn, progress=progress)
                    v = s["visibility_min"]
                    results.append({"video": path.name, "#": vid, "minutes": s["duration_min"],
                                    "moments": s["events"], "skipped": f"{s['skipped_share']:.0%}",
                                    "usable min": round(v["clear"] + v["murky"], 1)})
                except Exception as e:
                    results.append({"video": path.name, "#": None, "minutes": None, "moments": None,
                                    "skipped": f"failed: {e}", "usable min": None})
            bar.progress(1.0, text="All done")
            st.success(f"Processed {len(waiting)} videos.")
            first = next((r["#"] for r in results if r["#"] is not None), None)
            if first is not None:
                st.button("Start reviewing", type="primary", icon=":material/arrow_forward:",
                          on_click=go_to, args=(PAGES[1], first), key="review_batch")
            st.dataframe(pd.DataFrame(results), hide_index=True)

    if st.button("Process video", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        try:
            vid, s = process_video(video_path, site=site, use_ai=use_ai, conn=conn,
                                   survey_date=survey_day.isoformat(),
                                   progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m))
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.stop()
        v = s["visibility_min"]
        st.session_state["current_video"] = vid
        st.success(f"Done. Video #{vid} is ready to review.")
        st.button("Review this video now", type="primary", icon=":material/arrow_forward:",
                  on_click=go_to, args=(PAGES[1], vid), key="review_now")
        c = st.columns(4)
        c[0].metric("Video length", f"{s['duration_min']} min", border=True)
        c[1].metric("Moments to review", s["events"], f"{s['event_min']} min of video", delta_color="off",
                    border=True)
        c[2].metric("Footage skipped", f"{s['skipped_share']:.0%}", border=True)
        c[3].metric("Usable minutes", f"{v['clear'] + v['murky']:.1f}",
                    f"{v['unusable'] + v['blocked']:.1f} min too murky or blocked", delta_color="off",
                    border=True)
        if s.get("camera_events"):
            st.caption(f"Also skipped: {s['camera_events']} moments where the camera was being put in or taken out.")
        st.markdown("**Water clarity, minute by minute**")
        st.altair_chart(clarity_chart(db.query(conn, "SELECT * FROM segments WHERE video_id = ?", (vid,))),
                        width="stretch")
        st.caption(f"Estimated energy used: {describe_kwh(s['energy_kwh'])}")
        if s["ai_used"]:
            st.caption(f"AI found {s['detections']} boxes (fish, crabs, shrimp). " + "; ".join(
                f"{species.FLAG_TEXT.get(k, k)}: {v}" for k, v in s["flags"].items()))


# ---------------------------------------------------------------- 2 · review
elif page.startswith("2"):
    top = st.columns([4, 3, 3], vertical_alignment="bottom")
    with top[0]:
        video = pick_video("review_video")
    vid = int(video.id)
    if not st.session_state.get(f"camera_checked_{vid}"):  # for videos processed before this check existed
        mark_camera_handling(conn, vid, float(video.duration_s))
        mark_scenery(conn, vid)
        st.session_state[f"camera_checked_{vid}"] = True
    events = db.query(conn, "SELECT * FROM events WHERE video_id = ? ORDER BY start_s", (vid,))
    if events.empty:
        st.info("No moments in this video: seaSEArch found no movement.")
        st.stop()

    camera = int((events.status == "camera").sum())
    to_review = events[events.status != "camera"]
    done = int((to_review.status != "pending").sum())
    top[1].progress(done / max(1, len(to_review)), text=f"{done} of {len(to_review)} moments reviewed")
    only_pending = top[2].checkbox(
        "Only moments to review", value=True,
        help="Untick to see every moment again" + (
            f", including the {camera} where the camera was being put in or taken out (skipped "
            "automatically, like the first and last minute in Coastal Action's protocol)" if camera else "") + ".")
    view = events[events.status == "pending"] if only_pending else events
    if view.empty:
        undo_line(vid)
        st.success("All moments reviewed.")
        st.button("Open the report", type="primary", icon=":material/arrow_forward:",
                  on_click=go_to, args=(PAGES[2],))
        st.stop()

    key = f"pos_{vid}_{only_pending}"
    jump = st.session_state.pop("jump_to_event", None)  # after Undo: back to that moment
    if jump in list(view.id):
        st.session_state[key] = list(view.id).index(jump)
    pos = min(st.session_state.get(key, 0), len(view) - 1)
    ev = view.iloc[pos]
    site_df = load_site(video.site)
    dets = db.query(conn, "SELECT * FROM detections WHERE event_id = ? "
                    "AND IFNULL(flag, '') NOT IN ('scenery', 'false_alarm')",
                    (int(ev.id),))  # hidden: scenery (pole, rock) and things that aren't animals
    moments = review_moments(ev.start_s, ev.end_s, ev.peak_s)  # the busiest moment, then 3 across the event
    path = str(video.path)
    moved = clips.read_boxes(ev.boxes)

    def next_moment():
        """After finishing a moment. With 'Only moments to review' on, it leaves the list by itself."""
        if not only_pending:
            st.session_state[key] = min(pos + 1, len(view) - 1)

    # One line: where you are, and what this moment is like
    quality = "clear" if ev.visibility >= config.CLEAR_SCORE else \
        "murky" if ev.visibility >= config.MURKY_SCORE else "too murky to identify"
    who, reasons = difficulty.who_should_review(float(ev.visibility), moved)
    nav = st.container(horizontal=True, vertical_alignment="center")
    if nav.button("Previous", disabled=pos == 0, shortcut="PageUp", icon=":material/arrow_back:"):
        st.session_state[key] = pos - 1
        st.rerun()
    if nav.button("Next", disabled=pos >= len(view) - 1, shortcut="PageDown", icon=":material/arrow_forward:"):
        st.session_state[key] = pos + 1
        st.rerun()
    nav.markdown(f"**Moment {pos + 1} of {len(view)}** · {fmt_time(ev.start_s)} to {fmt_time(ev.end_s)}"
                    f" · footage **{quality}** ({ev.visibility:.0f}/100) · {difficulty.badge(who, reasons)}")

    arrival = st.session_state.get("new_arrival")
    if arrival and arrival["video_id"] == vid:
        show_new_arrival(arrival, site_df)

    left, right = st.columns(2, gap="medium")
    with left:
        switches = st.container(horizontal=True, gap="small")
        enhanced = switches.toggle("Enhance", value=bool(ev.visibility < config.CLEAR_SCORE),
                                      key=f"enh_{ev.id}",
                                      help="Clears up murky water. On by default for murky moments.")
        compare = switches.toggle("Side by side", key="compare",
                                     help="The original and the enhanced picture next to each other.")
        playing = switches.toggle("Play video", key="play_clip",
                                     help="This moment as a short clip, enhanced if Enhance is on. "
                                          "Each clip is made once, then saved.")
        if ev.status == "camera":
            st.markdown(":gray-badge[Camera being put in or taken out: skipped automatically]")

        # The busiest moment, with boxes: the AI's boxes if it ran, otherwise where seaSEArch saw movement
        at_peak = dets[(dets.frame_s - moments[0]).abs() < 0.05] if not dets.empty else dets
        caption = f"Busiest moment at {fmt_time(ev.peak_s)}" + (
            ". Yellow boxes: what the AI found" if not at_peak.empty else
            ". Orange boxes: where seaSEArch saw movement (no AI)" if moved else "")

        def busiest_view(enhance_it):
            frame = review_picture(vid, path, moments[0], enhance_it)
            if frame is None:
                return None
            if not at_peak.empty:
                return draw_boxes(frame, at_peak, scale=frame.shape[1] / full_width(path))
            return clips.draw_movement(frame, moved)

        clip = None
        if playing:  # the clip takes the big picture's place, so nothing moves further down
            clip = get_clip(event_clip_job(video, ev, enhanced),
                            "Making the clip (a few seconds; longer for 4K video)...")
            if clip:
                play_clip(clip)
                st.caption(("Enhanced clip. " if enhanced else "Original colours. ")
                           + "Orange boxes: where seaSEArch saw movement.")
            else:
                st.caption("This computer can't write video clips, so here's the picture.")
            if pos + 1 < len(view):  # get the next moment's clip ready while you look at this one
                clip_maker().request(*event_clip_job(video, view.iloc[pos + 1], enhanced))
        if clip is None:
            if compare:
                pair = st.columns(2)
                for col, (enh, title) in zip(pair, [(False, "Original"), (True, "Enhanced by seaSEArch")]):
                    picture = busiest_view(enh)
                    if picture is not None:
                        col.image(picture, channels="BGR", caption=title, width="stretch")
                st.caption(caption)
            else:
                picture = busiest_view(enhanced)
                if picture is None:
                    st.caption("(picture not available)")
                else:
                    st.image(picture, channels="BGR", caption=caption, width="stretch")

        cols = st.columns(3)
        for col, t in zip(cols, moments[1:]):
            frame = review_picture(vid, path, t, enhanced)
            if frame is None:
                col.caption("(picture not available)")
            else:
                col.image(frame, channels="BGR", caption=fmt_time(t), width="stretch")

    with right:
        undo_line(vid)

        suggestion, suggested_n = None, 1
        if not dets.empty:
            counts = counting.suggested_counts(dets)
            summary = (dets.assign(species_1=dets.species_1.fillna(""))
                       .groupby(["species_1", "flag"])
                       .agg(boxes=("id", "count"), best_score=("score_1", "max"))
                       .reset_index().sort_values("boxes", ascending=False))
            lines = []
            for r in summary.itertuples():
                name = species.common_name(site_df, r.species_1) if r.species_1 else "Fish"
                n = counts.get((int(ev.id), r.species_1), 1)
                score = f", score {r.best_score:.2f}" if r.species_1 else ""
                tag = species.badge(species.species_category(site_df, r.species_1)) if r.species_1 else ""
                lines.append(f"- {name} {tag}: up to {n} at once{score}. "
                             f"{species.FLAG_TEXT.get(r.flag, r.flag)}")
            st.markdown("**AI suggestions** (local, not checked yet)  \n" + "\n".join(lines))
            top_species = summary.iloc[0].species_1
            suggestion = top_species or None
            suggested_n = counts.get((int(ev.id), top_species), 1)
            crops = [p for p in dets.crop_path.head(6) if p]
            if crops:
                st.image([str(config.full_path(p)) for p in crops], width=56)

        saved = db.query(conn, "SELECT * FROM labels WHERE event_id = ?", (int(ev.id),))
        for r in saved.itertuples():
            row = st.columns([4, 1], vertical_alignment="center")
            tag = species.badge(species.species_category(site_df, r.species))
            row[0].markdown(f"Saved: {species.common_name(site_df, r.species)} x{r.count} {tag}")
            if row[1].button("Remove", key=f"del_{r.id}", width="stretch"):
                db.delete_label(conn, int(r.id))
                if (st.session_state.get("last_saved") or {}).get("label_id") == int(r.id):
                    del st.session_state["last_saved"]
                st.rerun()

        count_key = f"count_{ev.id}"

        def pick(name, proof=None, note=""):
            """Save a species for this moment, and move on unless the reviewer turned that off."""
            finish = st.session_state.get("auto_next", True)
            save_species(ev, video, site_df, name, int(st.session_state.get(count_key) or suggested_n),
                         moments[0] if proof is None else proof, note, finish)
            if finish:
                next_moment()

        c = st.columns([1, 1.3], vertical_alignment="bottom")
        c[0].number_input("How many at once?", min_value=1, value=int(suggested_n), key=count_key,
                          help="The most of this species you can see at once in this moment (MaxN).")
        c[1].toggle("Next moment after saving", value=st.session_state.get("auto_next_pref", True),
                    key="auto_next",
                    on_change=lambda: st.session_state.update(auto_next_pref=st.session_state["auto_next"]),
                    help="Turn off for a moment with more than one species: save each one, then click Done.")

        if site_df is not None:
            help_me_identify(int(ev.id), video.site, site_df, dets, set(saved.species), pick)

    with left:  # what to do with the whole moment, right under its pictures
        b = st.container(horizontal=True)
        if b.button("Nothing here", shortcut="Mod+Shift+X", icon=":material/block:",
                    help="No animal in this moment (weed, sediment, light flicker...): go to the next one"):
            db.set_event_status(conn, int(ev.id), "false_alarm")
            next_moment()
            st.rerun()
        if b.button("Done", disabled=saved.empty, shortcut="Mod+Shift+Enter", icon=":material/done_all:",
                    help="Every species in this moment is saved: go to the next one"):
            db.set_event_status(conn, int(ev.id), "reviewed")
            next_moment()
            st.rerun()
        listed = site_df[site_df.status != "not_fish"] if site_df is not None else pd.DataFrame()
        options = list(listed.scientific_name) if not listed.empty else []
        proof_options = []  # busiest moment first, skipping repeats of the same second
        for t in moments:
            if fmt_time(t) not in [fmt_time(x) for x in proof_options]:
                proof_options.append(t)
        with st.expander("Type the name instead, or add a note", expanded=site_df is None):
            with st.form(f"label_{ev.id}", clear_on_submit=True, border=False):
                label_name = lambda n: f"{species.common_name(site_df, n)} ({n})"  # noqa: E731
                # starts empty (or on the AI's suggestion), so nobody saves the wrong species by accident
                choice = st.selectbox("Species", options, key=f"species_{ev.id}", format_func=label_name,
                                      index=options.index(suggestion) if suggestion in options else None,
                                      placeholder="Start typing a name")
                other = st.text_input("Not on the list? Type it here", key=f"other_{ev.id}")
                proof = st.radio("Which picture shows that many?", proof_options, horizontal=True,
                                 format_func=lambda t: ("busiest, " if t == moments[0] else "") + fmt_time(t))
                note = st.text_input("Note (optional)")
                if st.form_submit_button("Save this species", type="primary", shortcut="Mod+Enter"):
                    name = (other or "").strip() or choice
                    if not name:
                        st.error("Pick a species or type one.")
                    else:
                        pick(name, proof=proof, note=note)
                        st.rerun()


# ---------------------------------------------------------------- 3 · report
elif page.startswith("3"):
    st.title("Survey report")
    video = pick_video("report_video")
    vid = int(video.id)
    events = db.query(conn, "SELECT * FROM events WHERE video_id = ? ORDER BY start_s", (vid,))
    segments = db.query(conn, "SELECT * FROM segments WHERE video_id = ? ORDER BY start_s", (vid,))
    labels = db.query(conn, "SELECT * FROM labels WHERE video_id = ?", (vid,))
    dets = db.query(conn, "SELECT * FROM detections WHERE video_id = ?", (vid,))
    site_df = load_site(video.site)
    minutes = minutes_by_label(segments.to_dict("records"))

    st.download_button("Download report (one file, with pictures)", build_report_html(conn, vid),
                       f"seasearch_report_{video.filename.rsplit('.', 1)[0]}.html", mime="text/html",
                       type="primary")
    st.caption("Opens in any browser, even offline. To make a PDF: open it and press Ctrl/Cmd+P.")

    c = st.columns(5)
    c[0].metric("Video length", f"{video.duration_s / 60:.1f} min", border=True)
    found = events[events.status != "camera"] if not events.empty else events
    c[1].metric("Moments found", len(found), border=True)
    c[2].metric("Reviewed", int((found.status != "pending").sum()) if not found.empty else 0, border=True)
    c[3].metric("Usable minutes", f"{minutes['clear'] + minutes['murky']:.1f}", border=True)
    c[4].metric("Energy (estimated)", describe_kwh(video.energy_kwh).split(" (")[0], border=True)
    if not events.empty:
        whos = [difficulty.who_should_review(float(e.visibility), clips.read_boxes(e.boxes))[0]
                for e in found.itertuples()]
        st.caption(f"{whos.count('volunteer')} of {len(whos)} moments are volunteer-friendly. "
                   f"{whos.count('expert')} need an expert (murky water, many fish at once, or tiny fish).")

    st.subheader("MaxN by species")
    st.caption("MaxN = the most individuals of a species seen at the same time, confirmed by a person. "
               "It never counts a fish twice.")
    table = counting.maxn_table(labels, events)
    if table.empty:
        st.info("No species confirmed yet. Review moments on page 2.")
    else:
        table.insert(0, "common_name", [species.common_name(site_df, n) for n in table.species])
        table["category"] = [species.species_category(site_df, n) for n in table.species]
        table["unexpected"] = [species.is_unexpected(site_df, n) for n in table.species]
        flagged = table[table.unexpected]
        if not flagged.empty:
            st.warning("Confirmed species that aren't expected here or are on the watch list: "
                       + ", ".join(flagged.common_name) + ". Get an expert to double-check.")
        at_risk = table[table.category == "at_risk"]
        if not at_risk.empty:
            st.info("Species at risk confirmed: " + ", ".join(at_risk.common_name)
                    + ". Share exact locations carefully.")
        for r in table.itertuples():
            row = st.container(border=True).columns([1, 3])
            with row[0]:
                frame = review_picture(vid, str(video.path), round(float(r.time_s), 2))
                fallback = r.thumb_path if isinstance(r.thumb_path, str) else None
                show_image(frame if frame is not None else fallback, width=220)
            sci = f" ({r.species})" if r.common_name != r.species else ""
            row[1].markdown(f"**{r.common_name}**{sci} {species.badge(r.category)}  \n"
                            f"MaxN **{r.maxn}**, proof frame at {fmt_time(r.time_s)} (event #{r.event_id})")

    if video.site:
        firsts = [n for n in sightings.first_seen_in_video(conn, video.site, vid)
                  if sightings.needs_alert(site_df, n)]
        if firsts:
            st.subheader("🆕 First sightings at this site")
            st.markdown(" · ".join(f"**{species.common_name(site_df, n)}** "
                                   f"{species.badge(species.species_category(site_df, n))}" for n in firsts))
            st.caption("Confirmed at this site for the first time in this survey. "
                       "The Sightings page has the full history.")

    st.subheader("Footage quality")
    st.caption("A relative score from contrast and sharpness, tuned on your own clips. It's not a "
               "turbidity measurement. Zero fish in unusable minutes doesn't mean zero fish were there.")
    q = st.columns(4)
    for col, key in zip(q, ["clear", "murky", "unusable", "blocked"]):
        col.metric(f"{key.capitalize()} minutes", f"{minutes[key]:.1f}", border=True)
    if not segments.empty:
        st.altair_chart(clarity_chart(segments), width="stretch")

    if not events.empty:
        st.subheader("Activity over time")
        per_minute = (events.assign(minute=(events.start_s // 60).astype(int))
                      .groupby("minute").size().rename("events"))
        st.bar_chart(per_minute)

    if not dets.empty:
        st.subheader("AI flags (before review)")
        st.write({species.FLAG_TEXT.get(k, k): int(v) for k, v in dets.flag.value_counts().items()})
        unexpected = dets[dets.flag == "unexpected"]
        if not unexpected.empty:
            st.warning("The AI thinks these might be unexpected or watch-list species. Get an expert to check:")
            st.image([str(config.full_path(p)) for p in unexpected.crop_path.head(8)], width=110,
                     caption=[species.common_name(site_df, n) for n in unexpected.species_1.head(8)])

    st.subheader("Download")
    d = st.columns(4)
    if not table.empty:
        d[0].download_button("MaxN (CSV)", table.to_csv(index=False), f"maxn_video_{vid}.csv")
    d[1].download_button("Confirmed labels (CSV)", labels.to_csv(index=False), f"labels_video_{vid}.csv")
    d[2].download_button("Events (CSV)", events.to_csv(index=False), f"events_video_{vid}.csv")
    d[3].download_button("Footage quality (CSV)", segments.to_csv(index=False), f"visibility_video_{vid}.csv")


# ---------------------------------------------------------------- 4 · sightings
elif page.startswith("4"):
    st.title("Sightings and new arrivals")
    st.write("Every species a person confirmed, by site and survey date. It's an early warning for "
             "new arrivals: species that aren't usual at a site, tropical strays, invasives, and "
             "species at risk. Built only from confirmed reviews, no AI.")
    sites = sorted(s for s in db.query(conn, "SELECT DISTINCT site FROM videos").site if s)
    if not sites:
        st.info("No processed videos with a site species list yet. Start on page 1.")
        st.stop()
    site = st.selectbox("Site", sites)
    site_df = load_site(site)
    table = sightings.summary(conn, site, site_df)
    arrivals = sightings.new_arrivals(table)
    old, never = sightings.not_seen_lately(table, site_df)
    surveys = int(db.query(conn, "SELECT COUNT(*) AS n FROM videos WHERE site = ?", (site,)).n.iloc[0])
    m = st.columns(4)
    m[0].metric("Surveys", surveys, border=True)
    m[1].metric("Species confirmed", len(table), border=True)
    m[2].metric("New arrivals and watch list", len(arrivals), border=True)
    m[3].metric("Not seen lately", len(old), border=True)

    st.subheader("🆕 New arrivals and watch-list species")
    if arrivals.empty:
        st.info("None yet. When a reviewer confirms a species that isn't usual here, it shows up here.")
    else:
        photos = cards.own_photos(conn)
        listed = set(site_df.scientific_name) if site_df is not None else set()
        for r in arrivals.itertuples():
            box = st.container(border=True).columns([1, 3])
            if r.species in photos:
                box[0].image(str(config.full_path(photos[r.species])), width="stretch")
            else:
                row = site_df[site_df.scientific_name == r.species] if site_df is not None else pd.DataFrame()
                group = species.group_of(row.iloc[0].to_dict()) if not row.empty else "fish"
                box[0].image(cards.outline("", group), channels="BGR", width="stretch")
            box[1].markdown(f"**{r.common_name}** {species.badge(r.category)}  \n"
                            f"First seen **{r.first_seen}** · last seen {r.last_seen} · "
                            f"{r.surveys} survey(s), {r.sightings} sighting(s)")
            box[1].caption("On this site's species list." if r.species in listed else
                           "Not on this site's species list yet. Add it from the alert on the Review "
                           "page, or on the Species lists page.")

    st.subheader("Not seen lately")
    st.caption(f"Usual species for this site that were seen before, but not in the "
               f"{config.NOT_SEEN_DAYS // 365} years before the latest survey. They may be rarer now, or gone.")
    if old.empty:
        st.write("None.")
    else:
        st.dataframe(old[["common_name", "last_seen", "surveys"]].rename(columns={
            "common_name": "Species", "last_seen": "Last seen", "surveys": "Surveys"}), hide_index=True)
    if never:
        with st.expander(f"Usual species not confirmed in any survey yet ({len(never)})"):
            st.write(", ".join(never))

    st.subheader("All confirmed species at this site")
    if table.empty:
        st.write("Nothing confirmed yet.")
    else:
        st.dataframe(table[["common_name", "species", "first_seen", "last_seen", "surveys", "sightings"]]
                     .rename(columns={"common_name": "Species", "species": "Scientific name",
                                      "first_seen": "First seen", "last_seen": "Last seen",
                                      "surveys": "Surveys", "sightings": "Sightings"}), hide_index=True)
    st.download_button("Sightings (CSV)", sightings.site_sightings(conn, site).to_csv(index=False),
                       f"sightings_{site}.csv")


# ---------------------------------------------------------------- 5 · count frames (protocol)
elif page.startswith("5"):
    st.title("Count frames (MeanCount)")
    st.write("Coastal Action's video analysis protocol: equally spaced frames, each counted by a person. "
             "seaSEArch picks the frames, adds up your dots, and writes MeanCount and the count sheet. No AI.")
    top = st.columns([3, 1, 1.4], vertical_alignment="bottom")
    with top[0]:
        video = pick_video("count_video")
    top[1].text_input("Your initials", key="counter", help="Saved with every count.")
    vid = int(video.id)
    site_df = load_site(video.site)
    subs = db.query(conn, "SELECT * FROM subsamples WHERE video_id = ? ORDER BY n", (vid,))
    segments = db.query(conn, "SELECT * FROM segments WHERE video_id = ? ORDER BY start_s", (vid,))

    if subs.empty:
        events = db.query(conn, "SELECT * FROM events WHERE video_id = ?", (vid,))
        start, n_frames, bad = protocol.suggest_window(float(video.duration_s), segments.to_dict("records"),
                                                       events.to_dict("records"))
        st.subheader("Choose the analysis window")
        extra = max(0, round((start - config.SKIP_START_S) / 60))
        why = "It skips the first minute" + (
            f", plus {extra} more while the water was still cloudy" if extra else "") + (
            f", so all {n_frames} frames land in usable footage." if not bad else
            f". {bad} of the {n_frames} frames still land in cloudy footage.")
        st.info(f"Suggested start: **{fmt_time(start)}**. {why} You can change it.")
        c = st.columns(3)
        start_min = c[0].number_input("Start (minutes)", 0.0, float(video.duration_s) / 60, start / 60, step=0.5)
        n_frames = c[1].number_input("Frames to count", 1, 120, n_frames)
        every = c[2].number_input("Seconds between frames", 5, 600, config.SUBSAMPLE_EVERY_S)
        times = protocol.subsample_times(start_min * 60, int(n_frames), int(every), float(video.duration_s))
        if not segments.empty:
            st.altair_chart(clarity_chart(segments, times), width="stretch")
            st.caption("Bars: how clear the water is, minute by minute. Ticks: your count frames. "
                       "Grey band: the analysis window.")
        st.caption("Frames at " + ", ".join(fmt_time(t) for t in times))
        if st.button("Create the count frames", type="primary", disabled=not times):
            save_count_frames(vid, str(video.path), times, st.progress(0.0, text="Saving the frames..."))
            db.create_subsamples(conn, vid, times)
            st.rerun()
        st.stop()

    qc_mode = top[2].toggle("Quality check (blind recount)", key=f"qc_{vid}",
                            help="Count some frames again without seeing the first count. seaSEArch compares the "
                                 f"two counts (they should differ by less than {config.QC_MAX_DIFF_PCT:.0f}%).")
    if qc_mode and not subs.qc.any():
        if not subs.done.all():
            st.info(f"First count all {len(subs)} frames ({int(subs.done.sum())} done so far). "
                    "Then a second person recounts some of them here, without seeing the first dots.")
            st.stop()
        picks = protocol.pick_qc_frames(list(subs.n))
        st.write(f"Pick {len(picks)} frames at random for the blind second count (at least "
                 f"{config.QC_SHARE:.0%} of frames, and at least {config.QC_MIN_FRAMES}). Ideally a different "
                 "person counts them.")
        if st.button("Pick the frames", type="primary"):
            for n in picks:
                db.update_subsample(conn, int(subs[subs.n == n].id.iloc[0]), qc=1)
            st.rerun()
        st.stop()

    counting_area(video, subs, site_df, qc_mode)

    st.divider()
    st.subheader("MeanCount for this video")
    first = db.query(conn, "SELECT d.* FROM dots d JOIN subsamples s ON d.subsample_id = s.id "
                           "WHERE s.video_id = ? AND d.pass = 1", (vid,))
    sheet = protocol.spreadsheet(video.filename, subs, first, common=lambda n: species.common_name(site_df, n))
    counted = sheet[sheet.Counted == "yes"]
    reviewed = set(db.query(conn, "SELECT species FROM labels WHERE video_id = ?", (vid,)).species)
    richness = (set(first.species) - {"", None}) | reviewed
    c = st.columns(4)
    c[0].metric("Frames counted", f"{len(counted)} of {len(sheet)}", border=True)
    c[1].metric("MeanCount (fish per frame)", f"{protocol.mean_count(list(counted['Total Fish Count'])):.1f}",
                border=True)
    c[2].metric("Invertebrates per frame", f"{protocol.mean_count(list(counted['Invertebrate Count'])):.1f}",
                border=True)
    c[3].metric("Species richness", len(richness), border=True)
    st.caption("MeanCount = the mean number of fish per counted frame. Species richness = every species "
               "recorded for this video, in the count frames or on the Review page.")
    juveniles = st.checkbox("Juvenile fish present in this video", value=bool(video.juveniles == 1),
                            key=f"juv_{vid}")
    if juveniles != bool(video.juveniles == 1):
        db.set_juveniles(conn, vid, int(juveniles))
    st.dataframe(sheet, hide_index=True)
    st.download_button("Download the count sheet (CSV)", sheet.to_csv(index=False),
                       f"meancount_{video.filename.rsplit('.', 1)[0]}.csv", type="primary")
    if not segments.empty:
        st.markdown("**Where your frames are**")
        st.altair_chart(clarity_chart(segments, list(subs.time_s), height=190), width="stretch")

    qc = subs[subs.qc == 1]
    if not qc.empty:
        st.subheader("Quality check")
        second = db.query(conn, "SELECT d.* FROM dots d JOIN subsamples s ON d.subsample_id = s.id "
                                "WHERE s.video_id = ? AND d.pass = 2", (vid,))
        rows = []
        for q in qc.itertuples():
            a = protocol.frame_counts(first[first.subsample_id == q.id].to_dict("records"))["fish"]
            b = protocol.frame_counts(second[second.subsample_id == q.id].to_dict("records"))["fish"]
            diff = protocol.percent_difference(a, b)
            recounted = bool(protocol.text(q.qc_counter))
            result = ("waiting for the second count" if not recounted else
                      "OK" if diff < config.QC_MAX_DIFF_PCT else "Recount (preferably a different person)")
            rows.append({"Sub-sample #": int(q.n), "Video time": fmt_time(q.time_s), "First count": a,
                         "Second count": b if recounted else None,
                         "Difference": f"{diff:.0f}%" if recounted else "", "Result": result})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(f"Percent difference = |first - second| / their average. Under {config.QC_MAX_DIFF_PCT:.0f}% "
                   "is OK; otherwise recount the frame. Turn on the quality check switch above to count.")

    with st.expander("Start this video's count over"):
        if st.button("Delete this video's count frames and dots"):
            db.delete_subsamples(conn, vid)
            st.rerun()


# ---------------------------------------------------------------- species lists
elif page == "Species lists":
    st.title("Species lists")
    st.write("One list per survey site. Add new arrivals here. The species suggestions read this list "
             "every time, so nothing needs retraining. Tags in the report come from this list:")
    st.markdown(" ".join(species.badge(c) for c in species.CATEGORY_TEXT))
    names = [p.stem for p in sorted(config.SPECIES_DIR.glob("*.csv"))]
    choice = st.selectbox("Site", names + ["+ New site list"])
    if choice == "+ New site list":
        site_name = st.text_input("Name for the new site (letters, numbers and underscores)")
        table = pd.DataFrame({"common_name": pd.Series(dtype=str), "scientific_name": pd.Series(dtype=str),
                              "status": pd.Series(dtype=str), "expected": pd.Series(dtype=bool)}
                             | {col: pd.Series(dtype=str) for col in species.EXTRA_COLUMNS})
    else:
        site_name = choice
        table = species.load_species_list(config.SPECIES_DIR / f"{choice}.csv")
    edited = st.data_editor(
        table, num_rows="dynamic", hide_index=True, width="stretch", key=f"editor_{choice}",
        column_config={
            "common_name": st.column_config.TextColumn("Common name", required=True),
            "scientific_name": st.column_config.TextColumn("Scientific name", required=True),
            "status": st.column_config.SelectboxColumn("Status", options=sorted(species.STATUSES),
                                                       required=True),
            "expected": st.column_config.CheckboxColumn("Usually found here?", default=False),
            "group": st.column_config.SelectboxColumn("Group", options=species.GROUPS,
                                                      help="fish (or blank) / invertebrate: crabs, shrimp, "
                                                           "lobster, snails. Plants: set Status to not_fish."),
            "body": st.column_config.TextColumn("Body", help="long and thin / torpedo / tall and round / flat. "
                                                "Several? Separate with |. Blank = not sure."),
            "tail": st.column_config.TextColumn("Tail", help="forked / rounded / square"),
            "markings": st.column_config.TextColumn("Markings", help="spots / stripes / plain"),
            "tip": st.column_config.TextColumn("ID tip", help="One short tip, in your own words"),
            "photo": st.column_config.TextColumn("Photo file", help="A free-to-use photo in data/species_photos/"),
            "photo_credit": st.column_config.TextColumn("Photo credit"),
            "first_seen": st.column_config.TextColumn("First seen"),
            "evidence": st.column_config.TextColumn("Evidence"),
        })
    st.caption("Status: native, invasive, tropical_stray, at_risk, or not_fish (plants and seaweed "
               "the detector might mistake for fish). To add a species, use the + on the table or the "
               "empty row at the bottom. Body, tail, markings and tip feed the species cards on the "
               "Review page.")
    st.warning("Sample data: the body, tail, markings and tip we filled in are drafts for this prototype, "
               "not checked by a biologist. Replace them with your own before real use.")
    if st.button("Save list", type="primary"):
        clean = "".join(ch for ch in (site_name or "").strip().replace(" ", "_") if ch.isalnum() or ch == "_")
        if not clean:
            st.error("Give the site a name first.")
        else:
            try:
                n = species.save_species_list(edited, config.SPECIES_DIR / f"{clean}.csv")
            except ValueError as e:
                st.error(str(e))
            else:
                load_site.clear()
                st.success(f"Saved {n} species to {clean}.csv")
