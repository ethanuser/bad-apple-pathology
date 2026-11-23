"""Crops are similarities in the physical plane of a slide, in micrometres.

Each slide gives three matching channels at 4 µm per pixel, stored as transmitted
brightness so that empty glass is 255 and dense stain is dark:
0 density (total optical density), 1 hematoxylin, 2 eosin."""
from dataclasses import dataclass, asdict
from functools import lru_cache

import cv2
import numpy as np

from .inputs import ASSETS, SOURCES
from .slides import decode_level, glass_colour, macenko, optical_density

CHANNELS = ('density', 'hematoxylin', 'eosin')
ASPECT = 4 / 3
PANEL_SIZE = (1040, 780)
BASE_UM = 4.0          # nominal; each slide uses exactly twice its fine level's pitch
LEVELS = 7             # 4, 8, ... 256 µm
PREP = 'slide-prep-v4'


@dataclass(frozen=True)
class Pose:
    slide: int
    channel: int
    x_um: float
    y_um: float
    width_um: float
    angle_deg: float
    mirror: bool
    invert: bool
    smooth: float = .006     # Gaussian sigma of the silhouette field, as a fraction of crop width

    def record(self):
        return asdict(self)


class Slide:
    """One slide's matching planes, memory-mapped from a per-slide cache so a bank of
    a hundred or more slides needs little RAM: each pyramid level is a (3, H, W)
    uint8 .npy file beside the slide, read on demand by the operating system."""
    def __init__(self, spec):
        self.spec = spec
        self.key, self.tissue = spec['key'], spec['tissue']
        self.folder = ASSETS / self.key
        stamp = f"{PREP}:{spec['levels']['fine']['sha256']}"
        meta = self.folder / f'{PREP}.npz'
        if not (meta.exists() and str(np.load(meta)['stamp']) == stamp
                and all((self.folder / f'{PREP}-L{k}.npy').exists() for k in range(LEVELS))):
            self._build(meta, stamp)
        saved = np.load(meta)
        self.glass, self.tissue_mask, self.stains = saved['glass'], saved['tissue'], saved['stains']
        levels = [np.load(self.folder / f'{PREP}-L{k}.npy', mmap_mode='r') for k in range(LEVELS)]
        self.planes = levels[0]                    # (3, H, W) uint8 at base_um
        self.pyramid = [[lv[c] for lv in levels] for c in range(3)]
        self.fine_um = spec['levels']['fine']['um_per_px']
        self.base_um = 2*self.fine_um
        self.size_um = (self.planes.shape[2]*self.base_um, self.planes.shape[1]*self.base_um)
        self.widths = (700., 14000.)
        self.width_clip = (600., min(16000., .98*self.size_um[1]*ASPECT))

    def _build(self, meta, stamp):
        for old in self.folder.glob('slide-prep-*'):
            old.unlink()                           # caches from earlier preparation versions
        planes, glass, tissue = self._prepare()
        np.savez(meta, glass=glass, tissue=tissue, stains=self.stains, stamp=stamp)
        level = planes
        for k in range(LEVELS):
            np.save(self.folder / f'{PREP}-L{k}.npy', np.ascontiguousarray(level))
            level = np.stack([cv2.resize(c, (max(1, c.shape[1]//2), max(1, c.shape[0]//2)),
                                         interpolation=cv2.INTER_AREA) for c in level])

    def _prepare(self):
        rgb = self.fine()
        half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_AREA)
        glass = glass_colour(half)
        od = optical_density(half, glass)
        self.stains = macenko(od)
        with np.errstate(all='ignore'):   # Accelerate can raise spurious flags here
            conc = (od.reshape(-1, 3) @ np.linalg.inv(self.stains)).reshape(*half.shape[:2], 3)
        density = od.sum(axis=2)
        # Stained tissue is both dense and coloured; dust, pen and debris are grey.
        sat = cv2.cvtColor(half, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32)
        tissue = (cv2.GaussianBlur(density, (0, 0), 3) > .15) & (cv2.GaussianBlur(sat, (0, 0), 3) > 22)
        planes = []
        for value in (density, np.clip(conc[..., 0], 0, None), np.clip(conc[..., 1], 0, None)):
            top = np.percentile(value[tissue], 99.5) if tissue.any() else 1.
            planes.append(np.rint(255 - 255*np.clip(value/max(top, 1e-3), 0, 1)).astype(np.uint8))
        small = cv2.resize(tissue.astype(np.uint8), (tissue.shape[1]//4, tissue.shape[0]//4),
                           interpolation=cv2.INTER_AREA) > 0     # 16 µm
        return np.stack(planes), glass.astype(np.float32), small

    def fine(self):
        return fine_pyramid(self.key)[0]

    def overview(self):
        return decode_level((ASSETS / self.key / 'overview.dcm').read_bytes())

    def level_for(self, out_pitch):
        """Coarsest pyramid level no coarser than the output pixel."""
        k = int(np.clip(np.floor(np.log2(max(out_pitch, self.base_um)/self.base_um)), 0, LEVELS-1))
        return k, self.base_um * 2**k


@lru_cache(maxsize=2)
def fine_pyramid(key):
    """The decoded 2 µm level and its area-reduced halvings, for display crops. Two
    slides are kept, since consecutive frames mostly stay on one slide."""
    rgb = decode_level((ASSETS / key / 'fine.dcm').read_bytes())
    out = [rgb]
    while min(out[-1].shape[:2]) > 128:
        r = out[-1]
        out.append(cv2.resize(r, (r.shape[1]//2, r.shape[0]//2), interpolation=cv2.INTER_AREA))
    return out


@lru_cache(maxsize=None)
def load_slides():
    return [Slide(s) for s in SOURCES['slides']]


@lru_cache(maxsize=24)
def unit_grid(w, h):
    x = ((np.arange(w, dtype=np.float32) + .5) / w - .5)[None, None, :]
    y = ((np.arange(h, dtype=np.float32) + .5) / h - .5)[None, :, None] / ASPECT
    return x, y


def maps(params, pitch, w, h, corners=False, origin=0.):
    """Output pixel centres to pixel coordinates of a plane with the given pitch (µm).
    Rows are x_um, y_um, width_um, angle_deg, mirror, measured from the top left edge
    of the slide."""
    p = np.asarray(params, dtype=np.float32)
    x, y, width, angle, mirror = (col[:, None, None] for col in p[:, :5].T)
    if corners:
        gx = np.array([[[-.5, .5, .5, -.5]]], dtype=np.float32)
        gy = np.array([[[-.5, -.5, .5, .5]]], dtype=np.float32) / ASPECT
    else:
        gx, gy = unit_grid(w, h)
    a = np.deg2rad(angle)
    c, s = width * np.cos(a), width * np.sin(a)
    u = gx * (1 - 2 * mirror)
    return (x + c*u - s*gy) / pitch - .5, (y + s*u + c*gy) / pitch - .5


def inside(params, size_um):
    x, y = maps(params, 1., 0, 0, corners=True)
    return ((x >= 0) & (x <= size_um[0]) & (y >= 0) & (y <= size_um[1])).all(axis=(1, 2))


def values(p):
    return [p.x_um, p.y_um, p.width_um, p.angle_deg, float(p.mirror)]


def sample(plane, pitch, params, w, h, interp=cv2.INTER_LINEAR, border=255):
    x, y = maps(params, pitch, w, h)
    out = cv2.remap(plane, x.reshape(-1, w), y.reshape(-1, w), interp,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return out.reshape(-1, h, w, plane.shape[2]) if plane.ndim == 3 else out.reshape(-1, h, w)


def field(slide, p, w, h):
    """The silhouette field: the channel sampled at the output scale and smoothed by
    a fixed fraction of the crop width, so the shape does not depend on magnification."""
    k, pitch = slide.level_for(p.width_um / w)
    img = sample(slide.pyramid[p.channel][k], pitch, [values(p)], w, h)[0].astype(np.float32)
    sigma = p.smooth * w
    return cv2.GaussianBlur(img, (0, 0), sigma) if sigma > .3 else img


def fields(slide, poses, w, h):
    return np.stack([field(slide, p, w, h) for p in poses])


def silhouette(slide, p, level, w, h, supersample=2):
    f = field(slide, p, w*supersample, h*supersample)
    white = ((f >= level) != p.invert).astype(np.float32) * 255
    return np.rint(cv2.resize(white, (w, h), interpolation=cv2.INTER_AREA)).astype(np.uint8)


def colour_crop(slide, p, w, h):
    """True-colour crop from the 2 µm level, or the area-reduced plane pyramid when
    the output pixel is coarser than 4 µm."""
    levels = fine_pyramid(slide.key)
    k = 0
    while k + 1 < len(levels) and p.width_um / w >= 2*slide.fine_um*2**k:
        k += 1
    rgb, pitch = levels[k], slide.fine_um*2**k
    g = tuple(float(v) for v in slide.glass)
    return sample(rgb, pitch, [values(p)], w, h, cv2.INTER_CUBIC, border=g)[0]


def footprint(slide, p, pitch):
    x, y = maps([values(p)], pitch, 0, 0, corners=True)
    return np.stack([x.ravel()+.5, y.ravel()+.5], axis=1).tolist()
