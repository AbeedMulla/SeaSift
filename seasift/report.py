"""One-file survey report: a single HTML file with the proof pictures inside it.

Opens in any browser with no internet, and prints to PDF (Ctrl/Cmd+P -> Save as PDF).
Built from the database, so it always matches what reviewers confirmed.
"""
import base64
import datetime
import html

import cv2

from . import config, counting, db, protocol, sightings, species
from .energy import describe_kwh
from .pipeline import thumb_folder
from .video import cached_frame
from .visibility import minutes_by_label

TAG_COLORS = {"green": "#2e7d32", "orange": "#e65100", "red": "#c62828", "violet": "#6a1b9a",
              "blue": "#1565c0", "gray": "#616161"}


def _fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _picture(video_id, video_path, t, thumb_path, width=420):
    """Proof picture as an embedded JPEG (so the report is one file). Uses the review picture
    saved during processing, so making the report never waits for the video."""
    try:
        frame = cached_frame(video_path, round(float(t), 2), thumb_folder(video_id), width=config.REVIEW_WIDTH)
    except Exception:
        frame = None
    if frame is None and isinstance(thumb_path, str):
        frame = cv2.imread(str(config.full_path(thumb_path)))
    if frame is not None and frame.shape[1] > width:
        frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])), interpolation=cv2.INTER_AREA)
    if frame is None:
        return "<em>picture not available</em>"
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    data = base64.b64encode(jpg.tobytes()).decode("ascii")
    return f'<img src="data:image/jpeg;base64,{data}" alt="proof frame at {_fmt_time(t)}">'


def _tag(category):
    color = TAG_COLORS[species.CATEGORY_COLOR[category]]
    return f'<span class="tag" style="background:{color}">{html.escape(species.CATEGORY_TEXT[category])}</span>'


def _meancount_html(conn, video, site_df=None):
    """The protocol (MeanCount) section, if frames were counted on the Count frames page."""
    subs = db.query(conn, "SELECT * FROM subsamples WHERE video_id = ? ORDER BY n", (int(video.id),))
    if subs.empty:
        return ""
    dots = db.query(conn, "SELECT d.* FROM dots d JOIN subsamples s ON d.subsample_id = s.id "
                          "WHERE s.video_id = ? AND d.pass = 1", (int(video.id),))
    sheet = protocol.spreadsheet(video.filename, subs, dots, common=lambda n: species.common_name(site_df, n))
    counted = sheet[sheet.Counted == "yes"]
    mean = protocol.mean_count(list(counted["Total Fish Count"]))
    juveniles = "yes" if video.juveniles == 1 else "no" if video.juveniles == 0 else "not recorded"
    rows = "".join(f"<tr><td>{r['Sub-sample #']}</td><td>{r['Video time']}</td><td>{html.escape(r['Visibility'])}</td>"
                   f"<td class='num'>{r['Total Fish Count']}</td><td>{r['Invertebrate Count']}</td>"
                   f"<td>{html.escape(str(r['Notes']))}</td></tr>" for _, r in counted.iterrows())
    return (f"<h2>MeanCount (Coastal Action protocol)</h2>"
            f"<p><strong>MeanCount: {mean:.1f} fish per frame</strong>, from {len(counted)} of {len(sheet)} "
            f"sub-sample frames counted by a person. Juveniles present: {juveniles}.</p>"
            "<table><tr><th>#</th><th>Time</th><th>Visibility</th><th>Fish</th><th>Invertebrates</th>"
            f"<th>Notes</th></tr>{rows}</table>")


def build_report_html(conn, video_id):
    """Return the full report for one video as an HTML string."""
    video = db.query(conn, "SELECT * FROM videos WHERE id = ?", (video_id,)).iloc[0]
    events = db.query(conn, "SELECT * FROM events WHERE video_id = ? ORDER BY start_s", (video_id,))
    segments = db.query(conn, "SELECT * FROM segments WHERE video_id = ? ORDER BY start_s", (video_id,))
    labels = db.query(conn, "SELECT * FROM labels WHERE video_id = ?", (video_id,))
    site_path = species.site_csv(video.site) if video.site else None
    site_df = species.load_species_list(site_path) if site_path else None

    minutes = minutes_by_label(segments.to_dict("records"))
    usable = minutes["clear"] + minutes["murky"]
    all_events, events = events, events[events.status != "camera"]  # camera handling was skipped
    reviewed = int((events.status != "pending").sum()) if not events.empty else 0
    event_min = float((events.end_s - events.start_s).sum()) / 60 if not events.empty else 0.0
    skipped = 1 - event_min * 60 / video.duration_s if video.duration_s else 0.0
    e = html.escape

    table = counting.maxn_table(labels, all_events)
    firsts = set(sightings.first_seen_in_video(conn, video.site, video_id)) if video.site else set()
    rows, notes = [], []
    for r in table.itertuples():
        category = species.species_category(site_df, r.species)
        name = species.common_name(site_df, r.species)
        sci = f"<br><span class='sci'>{e(r.species)}</span>" if name != r.species else ""
        first = r.species in firsts and category in sightings.ALERT_CATEGORIES
        new_tag = ' <span class="tag" style="background:#0d47a1">First sighting at this site</span>' if first else ""
        if first:
            notes.append(f"{e(name)}: first sighting at this site, in this survey.")
        rows.append(
            f"<tr><td><strong>{e(name)}</strong>{sci}</td>"
            f"<td>{_tag(category)}{new_tag}</td><td class='num'>{int(r.maxn)}</td>"
            f"<td>{_fmt_time(r.time_s)} (event #{int(r.event_id)})</td>"
            f"<td>{_picture(video_id, video.path, float(r.time_s), r.thumb_path)}</td></tr>")
        if category in {"new", "unusual", "tropical", "invasive"}:
            notes.append(f"{e(name)}: {e(species.CATEGORY_TEXT[category])}. Get an expert to double-check.")
        if category == "at_risk":
            notes.append(f"{e(name)}: species at risk. Share the exact location carefully.")
    species_html = (
        "<table><tr><th>Species</th><th>Tag</th><th>MaxN</th><th>Proof frame</th><th>Picture</th></tr>"
        + "".join(rows) + "</table>") if rows else "<p>No species confirmed yet.</p>"
    notes_html = ("<div class='notes'><strong>Needs attention</strong><ul>"
                  + "".join(f"<li>{n}</li>" for n in notes) + "</ul></div>") if notes else ""

    meancount_html = _meancount_html(conn, video, site_df)

    total = sum(minutes.values()) or 1
    bars = "".join(
        f"<div class='bar'><span class='lab'>{k}</span>"
        f"<span class='fill {k}' style='width:{100 * v / total:.0f}%'></span>"
        f"<span class='val'>{v:.1f} min</span></div>" for k, v in minutes.items())

    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>SeaSift report: {e(video.filename)}</title>
<style>
 body {{ font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; color:#1a1a1a;
        max-width: 1000px; margin: 24px auto; padding: 0 16px; line-height: 1.5; }}
 h1 {{ font-size: 24px; margin-bottom: 4px; }} h2 {{ font-size: 18px; margin-top: 28px; }}
 .meta {{ color:#555; font-size: 14px; }}
 .cards {{ display:flex; gap:12px; flex-wrap:wrap; margin-top:16px; }}
 .card {{ border:1px solid #ddd; border-radius:8px; padding:10px 14px; min-width:140px; }}
 .card .n {{ font-size:22px; font-weight:600; }} .card .l {{ font-size:13px; color:#555; }}
 table {{ border-collapse: collapse; width:100%; font-size:14px; }}
 th, td {{ border-bottom:1px solid #e3e3e3; padding:8px; text-align:left; vertical-align:top; }}
 td.num {{ font-size:20px; font-weight:600; }} .sci {{ font-style:italic; color:#555; font-size:13px; }}
 img {{ max-width:280px; border-radius:6px; }}
 .tag {{ color:#fff; border-radius:10px; padding:2px 8px; font-size:12px; white-space:nowrap; }}
 .notes {{ background:#fff4e5; border:1px solid #ffcc80; border-radius:8px; padding:8px 14px; margin:12px 0; }}
 .bar {{ display:flex; align-items:center; gap:8px; margin:4px 0; font-size:14px; }}
 .bar .lab {{ width:80px; }} .bar .val {{ width:80px; color:#555; }}
 .fill {{ height:14px; border-radius:4px; display:inline-block; min-width:2px; }}
 .clear {{ background:#2e7d32; }} .murky {{ background:#f9a825; }}
 .unusable {{ background:#c62828; }} .blocked {{ background:#424242; }}
 .method {{ font-size:13px; color:#444; }}
 @media print {{ body {{ margin:0; }} .card {{ break-inside: avoid; }} tr {{ break-inside: avoid; }} }}
</style></head><body>
<h1>Fish survey report</h1>
<div class="meta">Video: <strong>{e(video.filename)}</strong> · Site list: {e(video.site or "none")}
 · Survey date: {e(sightings.survey_day(video))}
 · Processed: {e(str(video.processed_at).replace("T", " "))} · Report made: {generated}<br>
 Made offline by SeaSift on this computer. Local AI used: {"yes" if video.ai_used else "no"}.</div>

<div class="cards">
 <div class="card"><div class="n">{video.duration_s / 60:.1f} min</div><div class="l">video length</div></div>
 <div class="card"><div class="n">{len(events)}</div><div class="l">moments with movement</div></div>
 <div class="card"><div class="n">{skipped:.0%}</div><div class="l">footage skipped</div></div>
 <div class="card"><div class="n">{reviewed} of {len(events)}</div><div class="l">moments reviewed</div></div>
 <div class="card"><div class="n">{usable:.1f} min</div><div class="l">usable footage</div></div>
</div>

<h2>Species and counts (MaxN)</h2>
{notes_html}
{species_html}

{meancount_html}

<h2>Footage quality</h2>
{bars}

<h2>How to read this report</h2>
<div class="method">
<p><strong>MaxN</strong> is the most individuals of a species seen at the same time anywhere in the video.
It never counts the same fish twice, but it can undercount when there are many fish. It's a number to
compare between surveys, not the total population.</p>
<p>Species and counts were <strong>confirmed by a person</strong>. Tags come from the site's species list.</p>
<p><strong>Footage quality</strong> is a relative score from contrast and sharpness. It isn't a turbidity
measurement. Zero fish in unusable minutes doesn't mean zero fish were there.</p>
<p>Estimated energy used to process this video: {e(describe_kwh(video.energy_kwh))}.</p>
</div>
</body></html>"""
