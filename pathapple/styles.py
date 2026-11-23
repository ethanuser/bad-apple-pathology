"""Display styles for a matched crop. All of them keep one threshold for the whole crop,
set at the matched level on the silhouette field, so the shape is the match itself.

brightfield  True H&E colour on glass. The figure side keeps saturated stain, the other
             side fades toward clean glass with faint texture.
fluorescence The same stains re-coloured as emission on black (hematoxylin blue,
             eosin magenta), like a virtual immunofluorescence view.
duotone      The black and white silhouette printed in two stain colours."""
import cv2
import numpy as np

from .geometry import colour_crop, field
from .slides import optical_density

GLASS = np.array([250, 248, 251], np.float32)
H_EMIT = np.array([70, 140, 255], np.float32)
E_EMIT = np.array([255, 60, 150], np.float32)


def side(slide, p, level, w, h, ramp=.5):
    """0 on the dark (stained) side of the threshold, 1 on the glass side, with a soft
    edge one or two output pixels wide plus a broad ramp."""
    f = field(slide, p, w, h)
    delta = f - level
    narrow = np.clip(.5 + delta / (2*max(1.5, 3*255/w)), 0, 1)
    broad = np.clip(.5 + .5*delta / np.where(delta < 0, max(level, 1), max(255-level, 1)), 0, 1)
    return (1-ramp*.3)*narrow + ramp*.3*broad


def brightfield(slide, p, level, w, h, strength=(1.35, .10)):
    rgb = colour_crop(slide, p, w, h)
    od = optical_density(rgb, slide.glass)
    s = side(slide, p, level, w, h)
    if p.invert:
        s = 1 - s
    gain = strength[0]*(1-s) + strength[1]*s
    out = GLASS * np.exp(-od*gain[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def fluorescence(slide, p, level, w, h, strength=(1.25, .08)):
    rgb = colour_crop(slide, p, w, h)
    od = optical_density(rgb, slide.glass)
    with np.errstate(all='ignore'):
        conc = np.clip((od.reshape(-1, 3) @ np.linalg.inv(slide.stains))[:, :2], 0, None).reshape(h, w, 2)
    # Stain emits, so the stained side (field below the level) is the lit one. With the
    # match not inverted the lit side must be the target's white: the glass side.
    lit = side(slide, p, level, w, h)
    if p.invert:
        lit = 1 - lit
    glow = (1 - np.exp(-conc*np.array([3.2, 2.4])))
    colour = glow[..., :1]*H_EMIT + glow[..., 1:]*E_EMIT
    colour = colour * (255/np.maximum(colour.max(axis=2, keepdims=True), 1)) ** .35
    gain = strength[1] + strength[0]*lit
    return np.clip(colour*gain[..., None], 0, 255).astype(np.uint8)


def brightfield_soft(slide, p, level, w, h):
    """Gentler tone: both sides keep visible tissue."""
    return brightfield(slide, p, level, w, h, (1.2, .38))


def adaptive(slide, p, level, w, h):
    """Brightfield when the match's dark side is stain (black figure on white),
    fluorescence when stain is the white side (white figure on black)."""
    return fluorescence(slide, p, level, w, h) if p.invert else brightfield(slide, p, level, w, h)


def duotone(slide, p, level, w, h, ink=(72, 28, 96), paper=(250, 240, 246)):
    s = side(slide, p, level, w, h, ramp=0)
    if p.invert:
        s = 1 - s
    ink, paper = np.array(ink, np.float32), np.array(paper, np.float32)
    return (ink*(1-s[..., None]) + paper*s[..., None]).astype(np.uint8)


STYLES = dict(brightfield=brightfield, soft=brightfield_soft, fluorescence=fluorescence,
              adaptive=adaptive, duotone=duotone)
