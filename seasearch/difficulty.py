"""Easy or hard? Who should review a moment, from things seaSEArch already measured. No AI.

Needs an expert when the water is murky, many things move at once (hard to count),
or the fish look tiny (too far away to identify). Otherwise a trained volunteer could do it,
and staff still check anything unusual they find (new arrivals, invasive, at risk).
"""
from . import config


def who_should_review(visibility, boxes):
    """Returns ("volunteer", []) or ("expert", [reasons]).

    visibility: the moment's footage-quality score (0-100)
    boxes: the orange movement boxes at its busiest moment, as (x, y, w, h) shares of the frame
    """
    reasons = []
    if visibility < config.CLEAR_SCORE:
        reasons.append("murky water")
    if len(boxes) > config.MANY_FISH:
        reasons.append(f"{len(boxes)} things moving at once")
    elif boxes and max(w * h for _, _, w, h in boxes) < config.TINY_FISH_SHARE:
        reasons.append("the fish look tiny (far away)")
    return ("expert", reasons) if reasons else ("volunteer", [])


def badge(who, reasons):
    """A coloured tag for Streamlit markdown."""
    if who == "volunteer":
        return ":green-badge[Volunteer-friendly]"
    return f":orange-badge[Needs an expert: {', '.join(reasons)}]"
