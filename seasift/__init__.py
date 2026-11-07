"""SeaSift: sift hours of murky survey video down to minutes of review.

Tier 0 (classical computer vision, no trained model): motion triage, visibility score,
enhancement, review, MaxN.
Tier 1 (optional, local AI): fish detector + BioCLIP species suggestions + flags.
"""
__version__ = "0.1.0"
