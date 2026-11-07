"""Make murky frames easier for a person to look at. Classical image processing, no trained model.

remove_cast: takes away most of the water's green, blue or yellow colour cast. It works in LAB
             colour space and leaves a little of the cast on purpose: removing all of it (the
             "gray world" method we used first) turned blue water pink on real underwater photos.
clahe:       boosts local contrast without blowing out bright areas.

Note: this is for reviewers' eyes. Don't assume it helps the AI detector too.
Test the detector on raw vs enhanced frames and keep whichever scores better.
"""
import cv2
import numpy as np

from . import config


def remove_cast(frame, strength=None):
    """Move the picture's average colour most of the way to neutral grey (1 = all the way)."""
    strength = config.CAST_STRENGTH if strength is None else strength
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    for channel in (1, 2):  # a = green to red, b = blue to yellow; 128 means no tint
        lab[:, :, channel] -= strength * (lab[:, :, channel].mean() - 128)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def clahe(frame, clip_limit=None, grid=8):
    """Contrast Limited Adaptive Histogram Equalization on the brightness channel."""
    clip_limit = config.CLAHE_CLIP if clip_limit is None else clip_limit
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)


def enhance(frame):
    return clahe(remove_cast(frame))
