"""Counting the way ecologists do it: MaxN.

MaxN = the most individuals of one species seen in a single frame.
It never counts the same fish twice, even if it swims out of view and comes back.
That's why we don't count with tracker IDs (they break in murky water).

MaxN here comes from what a PERSON confirmed (species + most seen at once + the proof
frame they picked), so it works without any AI model. The AI only pre-fills suggestions.
"""
import pandas as pd


def maxn_table(labels_df, events_df):
    """MaxN per species from confirmed labels, with the moment that proves it.

    labels_df: rows from the labels table (species, count, event_id, proof_s)
    events_df: rows from the events table (id, peak_s, thumb_path)
    time_s is the proof frame the reviewer picked, or the event's busiest moment if none.
    """
    columns = ["species", "maxn", "event_id", "time_s", "thumb_path"]
    if labels_df.empty:
        return pd.DataFrame(columns=columns)
    best = labels_df.sort_values("count", ascending=False).drop_duplicates("species")
    merged = best.merge(events_df[["id", "peak_s", "thumb_path"]],
                        left_on="event_id", right_on="id", how="left")
    proof = merged["proof_s"] if "proof_s" in merged else pd.Series(index=merged.index, dtype=float)
    merged["time_s"] = proof.fillna(merged["peak_s"])
    out = merged.rename(columns={"count": "maxn"})[columns]
    return out.sort_values(["maxn", "species"], ascending=[False, True]).reset_index(drop=True)


def suggested_counts(detections_df):
    """From AI detections: for each (event_id, species), the most boxes of that species in one frame.
    Used to pre-fill the count box in the review page. Species is "" in detector-only mode."""
    if detections_df.empty:
        return {}
    dets = detections_df.assign(species_1=detections_df["species_1"].fillna(""))
    per_frame = dets.groupby(["event_id", "frame_s", "species_1"]).size().reset_index(name="n")
    best = per_frame.groupby(["event_id", "species_1"])["n"].max().reset_index()
    return {(int(r.event_id), r.species_1): int(r.n) for r in best.itertuples()}
