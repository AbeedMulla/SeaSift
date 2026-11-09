"""Quick end-to-end test. Run it after every big change:  python tools/smoke_test.py

Writes to data/_smoke (its own database and pictures), so it never touches your real results.
The AI tier is tested with fake models, so this runs on any laptop.
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMOKE_OUT = ROOT / "data" / "_smoke"
shutil.rmtree(SMOKE_OUT, ignore_errors=True)
os.environ["SEASEARCH_OUT"] = str(SMOKE_OUT)  # must happen before importing seasearch
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from seasearch import cards, clips, config, counting, db, difficulty, protocol, sightings, species  # noqa: E402
from seasearch import ai as ai_module  # noqa: E402
from seasearch.pipeline import camera_handling, find_scenery, process_video, review_moments, unprocessed_videos  # noqa: E402
from seasearch.video import cached_frame  # noqa: E402
from seasearch.report import build_report_html  # noqa: E402

VIDEO = ROOT / "data" / "videos" / "test_murky.mp4"
results = []


def check(name, ok, detail=""):
    ok = bool(ok)
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


class FakeDetector:
    """Two fake fish that swim to a new spot every frame (boxes stuck in one spot count as scenery)."""
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def detect(self, frame):
        FakeDetector.calls += 1
        x, y = 20.0 + (FakeDetector.calls * 97) % 440, 20.0 + (FakeDetector.calls * 61) % 260
        return [(x, y, x + 80, y + 40, 0.9), (x + 100, y + 20, x + 180, y + 60, 0.8)]


class FakeSuggester:
    answer = "fish"  # what kind() says: set to "crab" to test finding crabs from movement

    def __init__(self, species_df, *args, **kwargs):
        self.crabs = list(species_df[species_df.group == "invertebrate"].scientific_name)

    def kind(self, image_path):
        return FakeSuggester.answer, 0.9

    def suggest(self, image_path, k=3, group="all"):
        if group == "invertebrate":
            return [(self.crabs[0], 0.7), (self.crabs[1], 0.2), (self.crabs[2], 0.05)]
        return [("Tautogolabrus adspersus", 0.8), ("Gadus morhua", 0.1), ("Chaetodon ocellatus", 0.05)]


def main():
    if not VIDEO.exists():
        import make_test_video
        make_test_video.main()
    conn = db.connect()

    # Tier 0: classical computer vision, no trained model
    vid, s = process_video(VIDEO, site="coastal_eelgrass_example", conn=conn, survey_date="2026-06-15")
    events = db.query(conn, "SELECT * FROM events WHERE video_id = ?", (vid,))
    truth = [(8, 12), (20, 25), (40, 44), (63, 67), (83, 87)]
    found = sum(any(e.start_s < b and e.end_s > a for e in events.itertuples()) for a, b in truth)
    check("triage finds fish passages", found >= 4, f"{found} of 5")
    check("triage skips most footage", s["skipped_share"] > 0.5, f"{s['skipped_share']:.0%} skipped")
    check("no false alarms from grass, sediment or murk changes", len(events) <= 5, f"{len(events)} events")
    labels = set(db.query(conn, "SELECT label FROM segments WHERE video_id = ?", (vid,)).label)
    check("visibility sees clear, murky, unusable and blocked",
          {"clear", "murky", "unusable", "blocked"} <= labels, ", ".join(sorted(labels)))
    check("thumbnails saved", all(config.full_path(p).exists() for p in events.thumb_path))
    moved = [clips.read_boxes(b) for b in events.boxes]
    check("movement boxes saved at each event's busiest moment (no AI)",
          all(moved) and all(0 <= v <= 1 for boxes in moved for box in boxes for v in box)
          and all(box[2] < 0.5 and box[3] < 0.5 for boxes in moved for box in boxes),
          f"{sum(len(b) for b in moved)} boxes in {len(moved)} events")

    # Enhanced clips: for the Review page (a browser format) and side by side for the pitch
    def orange_frames(path):
        cap, n, orange = cv2.VideoCapture(str(path)), 0, 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            n += 1
            orange += int(((f[:, :, 2] > 200) & (f[:, :, 1] > 90) & (f[:, :, 1] < 190) & (f[:, :, 0] < 80)).sum() > 50)
        return n, orange

    clip, kind = clips.make_clip(VIDEO, 8, 12, SMOKE_OUT / "clips" / "fish")
    n, orange = orange_frames(clip) if clip else (0, 0)
    check("enhanced event clip with orange movement boxes, in a format browsers play",
          clip is not None and clip.suffix in {".mp4", ".webm"} and n >= 50 and orange >= 10,
          f"{kind}, {n} frames, {orange} with boxes")
    pair, kind = clips.make_clip(VIDEO, 40, 44, SMOKE_OUT / "demo" / "murky", compare=True,
                                 fps_out=30, formats=clips.FOR_SLIDES)
    width = int(cv2.VideoCapture(str(pair)).get(cv2.CAP_PROP_FRAME_WIDTH)) if pair else 0
    check("demo clip: original and enhanced side by side", width == 2 * 640, f"{kind}, {width} px wide")
    outside, inside = Path("/tmp/elsewhere/pic.jpg"), ROOT / "data" / "pic.jpg"
    check("results can live outside the project (e.g. an external drive)",
          config.full_path(config.stored_path(outside)) == outside.resolve()
          and config.stored_path(inside) == str(Path("data") / "pic.jpg"))

    # Flags
    sp = species.load_species_list(ROOT / "data" / "species_lists" / "coastal_eelgrass_example.csv")
    cases = [
        (("Tautogolabrus adspersus", 0.8, 60), "known"),
        (("Chaetodon ocellatus", 0.8, 60), "unexpected"),
        (("Pterois volitans", 0.6, 60), "unexpected"),
        (("Tautogolabrus adspersus", 0.8, 10), "unidentifiable"),
        (("Zostera marina", 0.9, 60), "false_alarm"),
        (("Gadus morhua", 0.2, 60), "unsure"),
    ]
    for (name, score, vis), want in cases:
        got = species.decide_flag(name, score, vis, sp)
        check(f"flag {name} score {score} visibility {vis} -> {want}", got == want, got)

    # Species tags from the site list (no AI)
    lakes = species.load_species_list(ROOT / "data" / "species_lists" / "whitefish_lakes_example.csv")
    tags = [
        (sp, "Tautogolabrus adspersus", "normal"),
        (sp, "Chaetodon ocellatus", "tropical"),
        (sp, "Pterois volitans", "invasive"),
        (sp, "Salmo salar", "at_risk"),
        (sp, "Mystery fish", "new"),
        (sp, "Zostera marina", "not_fish"),
        (lakes, "Esox niger", "invasive"),
        (lakes, "Coregonus huntsmani", "at_risk"),
    ]
    wrong = {n: species.species_category(df, n) for df, n, want in tags
             if species.species_category(df, n) != want}
    check("species tags: normal / tropical / invasive / at risk / new / not a fish", not wrong, str(wrong))
    unusual = sp.copy()
    unusual.loc[unusual.scientific_name == "Gadus morhua", "expected"] = False
    check("species tags: native but not usually here -> unusual",
          species.species_category(unusual, "Gadus morhua") == "unusual")
    tmp_csv = SMOKE_OUT / "saved_list.csv"
    n_saved = species.save_species_list(sp, tmp_csv)
    again = species.load_species_list(tmp_csv)
    check("species list saves and loads back the same", n_saved == len(sp) and again.equals(sp))

    # MaxN from human labels
    first, second = int(events.id.iloc[0]), int(events.id.iloc[1])
    db.save_label(conn, first, vid, "Tautogolabrus adspersus", 1)
    db.save_label(conn, second, vid, "Tautogolabrus adspersus", 2)
    db.save_label(conn, second, vid, "Gadus morhua", 1)
    db.save_label(conn, second, vid, "Mystery fish", 1, proof_s=21.5)
    labels_df = db.query(conn, "SELECT * FROM labels WHERE video_id = ?", (vid,))
    table = counting.maxn_table(labels_df, events)
    cunner = table[table.species == "Tautogolabrus adspersus"].iloc[0]
    check("MaxN takes the most seen at once, not the total", cunner.maxn == 2 and cunner.event_id == second,
          f"MaxN {cunner.maxn}")
    mystery = table[table.species == "Mystery fish"].iloc[0]
    check("MaxN uses the proof frame the reviewer picked", abs(mystery.time_s - 21.5) < 1e-6,
          f"{mystery.time_s}")
    check("confirmed species off the list or on the watch list get flagged (no AI)",
          species.is_unexpected(sp, "Mystery fish") and species.is_unexpected(sp, "Chaetodon ocellatus")
          and not species.is_unexpected(sp, "Tautogolabrus adspersus"))

    # Question helper and species cards (no AI)
    long_thin = cards.filter_species(sp, {"body": "long and thin", "tail": cards.CANT_SEE, "markings": cards.CANT_SEE})
    kept = set(long_thin.common_name)
    check("species cards: answers hide species that don't match, blanks never hidden",
          {"American eel", "Northern pipefish", "Lined seahorse"} <= kept and "Cunner" not in kept
          and len(cards.filter_species(sp, {"body": cards.CANT_SEE})) == len(sp), f"{len(kept)} kept")
    check("species cards: answers saved as text",
          cards.describe({"body": "torpedo", "tail": "forked", "markings": cards.CANT_SEE})
          == "body: torpedo, tail: forked")
    order = cards.card_order(sp, {"Tautogolabrus adspersus": 3}, {"Gadus morhua": 0.9})
    check("species cards: AI suggestions first, then most seen here", order[:2] == ["Gadus morhua",
          "Tautogolabrus adspersus"], ", ".join(order[:3]))
    check("species cards: drawn outline when there's no photo", cards.outline("torpedo").shape == (180, 320, 3))

    # Proof pictures become card photos
    frame = cv2.imread(str(config.full_path(events.thumb_path.iloc[0])))
    label_id = db.save_label(conn, first, vid, "Microgadus tomcod", 1, traits="body: torpedo")
    db.set_label_photo(conn, label_id, cards.save_label_photo(frame, label_id))
    check("species cards: your own confirmed pictures are used first",
          "Microgadus tomcod" in cards.own_photos(conn)
          and db.query(conn, "SELECT traits FROM labels WHERE id = ?", (label_id,)).traits.iloc[0] == "body: torpedo")

    # Easy or hard: who should review a moment
    tests = [(30, [], "expert"), (80, [(0.1, 0.1, 0.1, 0.1)] * 5, "expert"),
             (80, [(0.1, 0.1, 0.02, 0.02)], "expert"), (80, [(0.1, 0.1, 0.1, 0.1)], "volunteer")]
    wrong = [(v, len(b)) for v, b, want in tests if difficulty.who_should_review(v, b)[0] != want]
    check("easy vs expert: murky water, many fish or tiny fish need an expert", not wrong, str(wrong))

    # Early warning: first sightings, new arrivals, not seen lately
    site = "coastal_eelgrass_example"
    mystery_id = int(db.query(conn, "SELECT id FROM labels WHERE species = 'Mystery fish'").id.iloc[0])
    check("survey date saved with the video",
          db.query(conn, "SELECT survey_date FROM videos WHERE id = ?", (vid,)).survey_date.iloc[0] == "2026-06-15")
    check("first sighting: alert for a species new to this site, not for usual ones",
          sightings.is_first_sighting(conn, site, "Mystery fish", mystery_id)
          and sightings.needs_alert(sp, "Mystery fish") and not sightings.needs_alert(sp, "Tautogolabrus adspersus"))
    old_vid = db.add_video(conn, VIDEO, site, 90, 15, False, survey_date="2022-05-01")
    old_event = {"start_s": 1, "end_s": 2, "peak_s": 1.5, "motion": 0.1, "visibility": 60, "thumb_path": None}
    db.add_events(conn, old_vid, [old_event])
    db.save_label(conn, old_event["id"], old_vid, "Pseudopleuronectes americanus", 1)
    table = sightings.summary(conn, site, sp)
    arrivals = set(sightings.new_arrivals(table).species)
    old, never = sightings.not_seen_lately(table, sp)
    check("new arrivals: species that aren't usual here, not the usual ones",
          "Mystery fish" in arrivals and "Tautogolabrus adspersus" not in arrivals, ", ".join(sorted(arrivals)))
    check("not seen lately: usual species last seen over 3 years ago",
          list(old.species) == ["Pseudopleuronectes americanus"] and "Atlantic cod" not in never and never,
          f"{len(old)} old, {len(never)} never seen")
    check("first sightings found per video", "Mystery fish" in sightings.first_seen_in_video(conn, site, vid))
    list_copy = SMOKE_OUT / "list_copy.csv"
    shutil.copy(ROOT / "data" / "species_lists" / f"{site}.csv", list_copy)
    species.add_to_list(list_copy, "Mystery fish", "tropical_stray", "2026-06-15", "test_murky.mp4 at 0:21")
    added = species.load_species_list(list_copy)
    row = added[added.scientific_name == "Mystery fish"]
    check("add a new arrival to the species list, with date and evidence",
          len(row) == 1 and row.first_seen.iloc[0] == "2026-06-15" and not row.expected.iloc[0]
          and species.species_category(added, "Mystery fish") == "tropical" and len(added) == len(sp) + 1)

    # Camera handling, saved review pictures, fish vs invertebrates
    long_video = 30 * 60
    check("camera handling: first/last minute and big movement near the ends skipped; short clips never",
          camera_handling({"peak_s": 20, "motion": 0.001}, long_video)
          and camera_handling({"peak_s": long_video - 100, "motion": 0.2}, long_video)
          and not camera_handling({"peak_s": long_video - 100, "motion": 0.01}, long_video)
          and not camera_handling({"peak_s": 900, "motion": 0.5}, long_video)
          and not camera_handling({"peak_s": 20, "motion": 0.5}, 90))
    check("review pictures: the busiest moment, then three across the moment",
          review_moments(10, 20, 14.5) == [14.5, 12.5, 15.0, 17.5])
    picture = cached_frame(VIDEO, 3.0, SMOKE_OUT / "cache_test", width=320)
    again = cached_frame("no_such_video.mp4", 3.0, SMOKE_OUT / "cache_test", width=320)
    check("pictures are decoded from the video once, then read from disk",
          picture is not None and again is not None and again.shape == picture.shape)
    coastal = species.load_species_list(species.site_csv("coastal_eelgrass_example"))
    crabs = species.names_in_group(coastal, "invertebrate")
    shown = cards.filter_species(coastal[coastal.status != "not_fish"], {"what": "invertebrate"})
    check("species lists tell fish from invertebrates (crabs, shrimp...)",
          "Carcinus maenas" in crabs and "Anguilla rostrata" not in crabs
          and "Zostera marina" not in species.names_in_group(coastal, "fish")
          and set(shown.scientific_name) == set(crabs), f"{len(crabs)} invertebrates")

    # Protocol mode: Coastal Action's MeanCount (equally spaced frames, counted by a person)
    settling = [{"label": "unusable", "start_s": 60, "end_s": 130}]
    start_s, n_frames, bad = protocol.suggest_window(30 * 60, settling, [])
    calm_start, calm_frames, _ = protocol.suggest_window(30 * 60, [], [])
    check("protocol: window skips the first minute, plus minutes still settling",
          calm_start == 60 and calm_frames == 15 and start_s == 180 and bad == 0, f"{calm_start} / {start_s}")
    times = protocol.subsample_times(60, 15, 60, 30 * 60)
    check("protocol: 15 frames, one every 60 s", len(times) == 15 and times[1] - times[0] == 60)
    dots = [{"x": 0.05, "y": 0.1, "kind": "fish", "species": "Gadus morhua"},
            {"x": 0.95, "y": 0.9, "kind": "unknown", "species": ""},
            {"x": 0.5, "y": 0.5, "kind": "invert", "species": "Carcinus maenas"}]
    counts = protocol.frame_counts(dots)
    grid = protocol.grid_counts(dots)
    check("protocol: fish, unknown fish and invertebrates counted separately, per grid cell",
          counts["fish"] == 2 and counts["unknown"] == 1 and counts["invertebrates"] == 1
          and len(grid[0]) == 6 and grid[0][0] == 1 and grid[2][5] == 1 and grid[1][2] == 1, str(counts))
    check("protocol: clicking a dot again finds it (to remove it)",
          protocol.nearest_dot([dict(d, id=i) for i, d in enumerate(dots)], 0.051, 0.101, 16 / 9)["id"] == 0
          and protocol.nearest_dot(dots, 0.3, 0.3, 16 / 9) is None)
    check("protocol: MeanCount and the 6% quality check",
          protocol.mean_count([36, 102, 0]) == 46 and protocol.percent_difference(100, 104) < 6
          and protocol.percent_difference(100, 110) > 6 and protocol.percent_difference(0, 0) == 0
          and len(protocol.pick_qc_frames(range(1, 16), seed=1)) == 6
          and len(protocol.pick_qc_frames(range(1, 61), seed=1)) == 6
          and len(protocol.pick_qc_frames(range(1, 4), seed=1)) == 3)
    db.create_subsamples(conn, vid, [20.0, 40.0])
    subs = db.query(conn, "SELECT * FROM subsamples WHERE video_id = ? ORDER BY n", (vid,))
    for d in dots:
        db.add_dot(conn, int(subs.id.iloc[0]), d["x"], d["y"], d["kind"], d["species"], "SK")
    db.update_subsample(conn, int(subs.id.iloc[0]), visibility="0.5-1.0 m", done=1, counter="SK")
    subs = db.query(conn, "SELECT * FROM subsamples WHERE video_id = ? ORDER BY n", (vid,))
    sheet = protocol.spreadsheet("test.mp4", subs, db.query(conn, "SELECT * FROM dots"))
    check("protocol: count sheet with Coastal Action's columns",
          list(sheet.columns[:7]) == ["Video ID", "Sub-sample #", "Video time", "Counted", "Visibility",
                                      "Total Fish Count", "Invertebrate Count"]
          and "Unknown/Unable to ID due to visibility" in sheet and sheet["Total Fish Count"].iloc[0] == 2
          and sheet["Gadus morhua"].iloc[0] == 1 and sheet["Visibility"].iloc[0] == "0.5-1.0 m")
    picture = protocol.draw(frame, dots)
    check("protocol: frame drawn with grid, dots and cell counts", picture.shape == frame.shape)

    # One-file report with embedded pictures
    report = build_report_html(conn, vid)
    out = SMOKE_OUT / "report.html"
    out.write_text(report)
    check("report file: species, MaxN, tags and pictures inside",
          "Cunner" in report and "MaxN" in report and "data:image/jpeg;base64," in report
          and "New: not on the list" in report and "Mystery fish" in report,
          f"{len(report) // 1024} KB")
    check("report file: survey date and first sightings at this site",
          "Survey date: 2026-06-15" in report and "First sighting at this site" in report)
    check("report file: MeanCount section from the counted frames",
          "MeanCount (Coastal Action protocol)" in report and "2.0 fish per frame" in report)
    check("batch: processed videos aren't queued again", VIDEO.resolve() not in
          [p.resolve() for p in unprocessed_videos(conn)])

    # AI tier with fake models (tests the wiring, not the models)
    still = [{"event_id": e, "x1": 10, "y1": 10, "x2": 60, "y2": 30} for e in (1, 2)]
    swim = [{"event_id": e, "x1": 100 + 90 * e, "y1": 50, "x2": 150 + 90 * e, "y2": 70} for e in (1, 2)]
    check("AI boxes stuck in one spot across moments are hidden as scenery", find_scenery(still + swim) == {0, 1})
    ai_module.FishDetector, ai_module.SpeciesSuggester = FakeDetector, FakeSuggester
    vid2, s2 = process_video(VIDEO, site="coastal_eelgrass_example", use_ai=True, conn=conn)
    dets = db.query(conn, "SELECT * FROM detections WHERE video_id = ?", (vid2,))
    check("AI tier saves detections with flags", len(dets) > 0 and set(dets.flag) <= set(species.FLAG_TEXT),
          f"{len(dets)} detections, flags {sorted(set(dets.flag))}")
    suggested = counting.suggested_counts(dets)
    check("suggested counts per event", max(suggested.values()) == 2, f"{suggested}")
    check("AI only asks about movement for crabs and shrimp (a 'fish' answer is the detector's job)",
          set(dets.source) == {"detector"})

    # Crabs and shrimp: the fish detector never boxes them, BioCLIP looks at what moved
    FakeSuggester.answer = "crab"
    vid_c, _ = process_video(VIDEO, site="coastal_eelgrass_example", use_ai=True, conn=conn)
    FakeSuggester.answer = "fish"
    dc = db.query(conn, "SELECT * FROM detections WHERE video_id = ?", (vid_c,))
    site = species.load_species_list(species.site_csv("coastal_eelgrass_example"))
    inverts = set(site[site.group == "invertebrate"].scientific_name)
    moved = dc[dc.source == "movement"]
    check("crabs found from movement, named from the site's invertebrates",
          len(moved) > 0 and set(moved.species_1) <= inverts, f"{len(moved)} from movement")

    # Detector only (species suggestions switched off): boxes and counts, people pick species
    config.SPECIES_SUGGESTIONS = False
    vid3, _ = process_video(VIDEO, site="coastal_eelgrass_example", use_ai=True, conn=conn)
    config.SPECIES_SUGGESTIONS = True
    d3 = db.query(conn, "SELECT * FROM detections WHERE video_id = ?", (vid3,))
    check("detector-only mode: no species guesses, still counts fish",
          len(d3) > 0 and set(d3.species_1) == {""} and set(d3.flag) <= {"fish", "unidentifiable"}
          and max(counting.suggested_counts(d3).values()) == 2)
    check("energy measured", s2["energy_kwh"] is None or s2["energy_kwh"] >= 0, str(s2["energy_kwh"]))

    # Real-footage scorer (tools/eval_aura.py) on a fake AURA-style folder: the fish at 8-12 s
    # is marked as frames 120-180 for v98, and as frames 60-90 of a half-rate soft_labels.csv for v99
    import eval_aura
    fake = SMOKE_OUT / "aura" / "AURA"
    fake.mkdir(parents=True)
    for name in ["v98.mp4", "v99.mp4"]:
        shutil.copy(VIDEO, fake / name)
    (fake / "consensus_events.csv").write_text(
        "scene,video,start_frame,end_frame\nscene_A,v98,120,180\nscene_B,v99,60,90\n")
    (fake / "soft_labels.csv").write_text("scene,video,frame_idx,soft_label\n" + "".join(
        f"scene_B,v99,{i},{1.0 if 60 <= i <= 90 else 0.0}\n" for i in range(675)))
    scored = {r["video"]: r for r in eval_aura.score_folder(SMOKE_OUT / "aura", quiet=True)}
    check("real-footage scorer (AURA) lines up people's labels with video time",
          len(scored) == 2 and all(r["found"] and abs(r["marked"][0] - 8.0) < 0.05 for r in scored.values())
          and scored["v99"]["scaled"] and not scored["v98"]["scaled"],
          ", ".join(f"{v}: {r['marked'][0]:.1f}-{r['marked'][1]:.1f} s" for v, r in sorted(scored.items())))

    # Click through the Streamlit app like a staff member would
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("SKIP  app tests (streamlit not installed)")
    else:
        app_flow(ai_video=vid2)

    print(f"\n{sum(results)} of {len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


def click(app, label):
    [b for b in app.button if b.label == label][0].click().run()


def app_flow(ai_video):
    from streamlit.testing.v1 import AppTest

    def ok(app):
        return not app.exception, str(app.exception[0].value) if app.exception else ""

    import datetime
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    check("app: process page opens", *ok(app))
    app.selectbox[0].set_value(VIDEO).run()  # the test clip, even when a big survey video sorts first
    app.date_input[0].set_value(datetime.date(2026, 8, 14)).run()
    click(app, "Process video")
    newest = db.query(db.connect(), "SELECT survey_date FROM videos ORDER BY id DESC LIMIT 1").survey_date.iloc[0]
    check("app: process a video, with its survey date", not app.exception and len(app.success) > 0
          and newest == "2026-08-14", newest)

    click(app, "Review this video now")
    check("app: 'Review this video now' opens the review page on that video",
          not app.exception and app.sidebar.radio[0].value == "2 · Review moments"
          and app.selectbox[0].value == int(db.query(db.connect(), "SELECT MAX(id) AS id FROM videos").id.iloc[0]))
    check("app: review page opens", *ok(app))
    box = [s for s in app.selectbox if s.key and s.key.startswith("species_")][0]
    check("app: the species box starts empty (no species saved by accident)", box.value is None, str(box.value))
    crabs = species.names_in_group(species.load_species_list(species.site_csv("coastal_eelgrass_example")),
                                   "invertebrate")

    def pills(label):
        return [p for p in app.button_group if p.label == label]

    def card_buttons():
        return [b for b in app.button if b.key and b.key.startswith("card_")]

    pills("What is it?")[0].set_value("invertebrate").run()
    crab_cards = len(card_buttons())
    check("app: 'What is it?' shows the invertebrate cards and hides the fish questions",
          not app.exception and crab_cards == min(6, len(crabs)) and not pills("Body")
          and any(t.label.startswith("Show") for t in app.toggle) == (len(crabs) > 6), f"{crab_cards} cards")
    check("app: the first cards can be picked with number keys",
          [b.proto.shortcut for b in card_buttons()[:3]] == ["1", "2", "3"]
          and card_buttons()[0].label == "This one", str([b.label for b in card_buttons()[:3]]))
    pills("What is it?")[0].set_value(None).run()

    def matching():  # from the caption "N of M species on this site's list match."
        text = [c.value for c in app.caption if "species on this site's list match" in c.value][0]
        return int(text.split(" of ")[0])

    def moment():  # "Moment 1 of 4 · 0:08 to 0:12 ..."
        return [m.value for m in app.markdown if m.value.startswith("**Moment ")][0].split(" · ")[1]

    before = matching()
    pills("Body")[0].set_value("long and thin").run()
    after = matching()
    check("app: question helper narrows the species cards", 0 < after < before, f"{before} -> {after} cards")
    first_moment = moment()
    app.number_input[0].set_value(2)
    [b for b in app.button if b.key and b.key.endswith("_Anguilla rostrata")][0].click().run()
    saved = db.query(db.connect(), "SELECT species, count, traits, photo_path, event_id FROM labels "
                                   "ORDER BY id DESC LIMIT 1")
    status = db.query(db.connect(), "SELECT status FROM events WHERE id = ?",
                      (int(saved.event_id.iloc[0]),)).status.iloc[0]
    check("app: one click on a card saves the species, with what the reviewer saw and a picture",
          not app.exception and saved.species.iloc[0] == "Anguilla rostrata" and saved["count"].iloc[0] == 2
          and saved.traits.iloc[0] == "body: long and thin" and isinstance(saved.photo_path.iloc[0], str))
    check("app: ...and moves on to the next moment", status == "reviewed" and moment() != first_moment
          and any("Saved American eel x2" in m.value for m in app.markdown), f"{first_moment} -> {moment()}")
    click(app, "Undo")
    gone = db.query(db.connect(), "SELECT COUNT(*) AS n FROM labels WHERE event_id = ?",
                    (int(saved.event_id.iloc[0]),)).n.iloc[0]
    check("app: Undo takes the save back and returns to that moment", not app.exception and gone == 0
          and moment() == first_moment, f"back at {moment()}")
    def save_eel():  # answers are per moment, so narrow the cards down again first
        pills("Body")[0].set_value("long and thin").run()
        [b for b in app.button if b.key and b.key.endswith("_Anguilla rostrata")][0].click().run()

    app.number_input[0].set_value(2)
    save_eel()
    [t for t in app.toggle if t.label == "Next moment after saving"][0].set_value(False).run()
    second_moment = moment()
    save_eel()
    check("app: with 'Next moment after saving' off, a save stays on the moment",
          moment() == second_moment and any("Saved: American eel x1" in m.value for m in app.markdown)
          and [b for b in app.button if b.key and b.key.endswith("_Anguilla rostrata")][0].disabled)
    app.text_input[0].set_value("Mystery fish 2")  # "Mystery fish" was already seen at this site above
    click(app, "Save this species")
    check("app: first sighting alert for a species new to this site",
          any("First sighting at this site" in m.value for m in app.markdown))
    click(app, "Not now")
    check("app: alert can be dismissed", not any("First sighting at this site" in m.value for m in app.markdown))
    click(app, "Done")
    check("app: Done moves on to the next moment", moment() != second_moment)
    click(app, "Nothing here")
    check("app: finish and dismiss events", *ok(app))

    app.sidebar.radio[0].set_value("3 · Report").run()
    check("app: report shows MaxN", any("MaxN **2**" in m.value for m in app.markdown))
    check("app: report warns about unexpected species", any("Mystery fish" in w.value for w in app.warning))
    check("app: report shows first sightings and volunteer-friendly moments",
          any("First sightings" in h.value for h in app.subheader)
          and any("volunteer-friendly" in c.value for c in app.caption))

    app.sidebar.radio[0].set_value("4 · Sightings").run()
    check("app: sightings page shows new arrivals", not app.exception
          and any("Mystery fish" in m.value for m in app.markdown))

    app.sidebar.radio[0].set_value("5 · Count frames (MeanCount)").run()
    check("app: count page suggests an analysis window", *ok(app))
    app.number_input[0].set_value(0.0)
    app.number_input[1].set_value(5)
    app.number_input[2].set_value(15).run()
    click(app, "Create the count frames")
    conn = db.connect()
    newest = int(db.query(conn, "SELECT MAX(id) AS id FROM videos").id.iloc[0])
    subs = db.query(conn, "SELECT * FROM subsamples WHERE video_id = ? ORDER BY n", (newest,))
    check("app: count frames created", not app.exception and len(subs) == 5 and subs.time_s.iloc[1] == 15,
          f"{len(subs)} frames")
    db.add_dot(conn, int(subs.id.iloc[0]), 0.2, 0.2, "fish", "", "SK")
    db.add_dot(conn, int(subs.id.iloc[0]), 0.4, 0.2, "fish", "", "SK")
    app.run()
    [r for r in app.radio if r.label == "Visibility"][0].set_value("1.0-1.5 m").run()
    click(app, "Save this frame, go to the next")
    saved = db.query(conn, "SELECT * FROM subsamples WHERE id = ?", (int(subs.id.iloc[0]),)).iloc[0]
    check("app: count a frame (dots, visibility) and see MeanCount",
          not app.exception and saved.done == 1 and saved.visibility == "1.0-1.5 m"
          and any(m.value == "2.0" for m in app.metric), str([m.value for m in app.metric]))
    app.toggle[0].set_value(True).run()
    waits = any("First count all 5 frames" in m.value for m in app.info)
    for sub_id in subs.id[1:]:  # the quality check starts once every frame has its first count
        db.update_subsample(conn, int(sub_id), done=1)
    app.run()
    click(app, "Pick the frames")
    picked = db.query(conn, "SELECT COUNT(*) AS n FROM subsamples WHERE video_id = ? AND qc = 1", (newest,))
    check("app: quality check waits for the first count, then picks frames for a blind recount",
          not app.exception and waits and int(picked.n.iloc[0]) == 5)
    db.add_dot(conn, int(subs.id.iloc[0]), 0.21, 0.2, "fish", "", "AB", pass_no=2)
    db.add_dot(conn, int(subs.id.iloc[0]), 0.41, 0.2, "fish", "", "AB", pass_no=2)
    app.run()
    click(app, "Save this frame, go to the next")
    results = [d.value for d in app.dataframe if "Result" in d.value]
    check("app: blind recount compared with the first count (2 and 2 fish: OK)",
          not app.exception and results and list(results[0].Result[:2]) == ["OK", "waiting for the second count"],
          str(results[0].head(2).to_dict("records")) if results else "no quality check table")

    app.sidebar.radio[0].set_value("2 · Review moments").run()
    app.selectbox[0].set_value(ai_video).run()
    check("app: review page with AI boxes and suggestions", *ok(app))
    app.toggle[0].set_value(True).run()
    check("app: enhanced frames with AI boxes", *ok(app))
    app.toggle[1].set_value(True).run()
    check("app: original and enhanced side by side", *ok(app))
    app.toggle[2].set_value(True).run()
    check("app: plays the enhanced event clip", not app.exception and len(app.get("video")) == 1
          and any("Enhanced clip" in c.value for c in app.caption))

    app.sidebar.radio[0].set_value("Species lists").run()
    check("app: species list editor opens", *ok(app))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
