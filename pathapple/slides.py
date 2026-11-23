"""Whole-slide images: tiled DICOM decoding, stain separation and the matching planes."""
import io

import cv2
import numpy as np
import pydicom
from PIL import Image
from pydicom.encaps import generate_frames


def decode_level(data):
    """Assemble one TILED_FULL pyramid level into an RGB array."""
    d = pydicom.dcmread(io.BytesIO(data))
    if d.get('DimensionOrganizationType', 'TILED_FULL') != 'TILED_FULL':
        raise ValueError('Only TILED_FULL slides are supported')
    R, C = d.Rows, d.Columns
    W, H = d.TotalPixelMatrixColumns, d.TotalPixelMatrixRows
    nx = -(-W // C)
    out = np.full((-(-H // R) * R, nx * C, 3), 255, np.uint8)
    if d.file_meta.TransferSyntaxUID.is_compressed:
        tiles = (np.asarray(Image.open(io.BytesIO(f)).convert('RGB'))
                 for f in generate_frames(d.PixelData, number_of_frames=d.NumberOfFrames))
    else:
        tiles = iter(d.pixel_array.reshape(-1, R, C, 3))
    for i, tile in enumerate(tiles):
        y, x = divmod(i, nx)
        out[y*R:(y+1)*R, x*C:(x+1)*C] = tile
    return out[:H, :W]


def optical_density(rgb, background=None):
    bg = np.asarray(background if background is not None else (245, 245, 245), np.float32)
    return -np.log(np.clip(rgb.astype(np.float32) / bg, 1/255, 1))


# Ruifrok and Johnston H&E vectors in optical density, third vector their cross product.
HE = np.array([[.650, .704, .286], [.072, .990, .105]], np.float32)
HE /= np.linalg.norm(HE, axis=1, keepdims=True)
STAINS = np.vstack([HE, np.cross(HE[0], HE[1]) / np.linalg.norm(np.cross(HE[0], HE[1]))])


def glass_colour(rgb):
    """Brightest common colour: the slide's empty glass."""
    small = rgb[::8, ::8].reshape(-1, 3).astype(np.float32)
    bright = small[small.sum(1) >= np.percentile(small.sum(1), 90)]
    return np.median(bright, axis=0)


def tissue_mask(rgb, background=None):
    od = optical_density(rgb, glass_colour(rgb) if background is None else background)
    small = cv2.GaussianBlur(od.sum(axis=2), (0, 0), 2)
    return small > .18


def separate(rgb, background=None):
    """Hematoxylin and eosin concentrations (optical density units)."""
    od = optical_density(rgb, background)
    conc = od.reshape(-1, 3) @ np.linalg.inv(STAINS)
    return conc[:, 0].reshape(rgb.shape[:2]), conc[:, 1].reshape(rgb.shape[:2]), od.sum(axis=2)


def macenko(od, beta=.15, alpha=1.):
    """Per-slide stain vectors (Macenko et al. 2009): the two extreme angles of tissue
    optical density in its principal plane. Hematoxylin is the one absorbing more red."""
    px = od.reshape(-1, 3)[::7]
    px = px[(px > beta).all(axis=1)]
    with np.errstate(all='ignore'):
        _, vecs = np.linalg.eigh(np.cov(px.T))
    plane = vecs[:, 1:3]
    proj = px @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, alpha), np.percentile(phi, 100-alpha)
    v = [plane @ np.array([np.cos(a), np.sin(a)]) for a in (lo, hi)]
    v = [x if x[0] >= 0 else -x for x in v]
    h, e = (v[0], v[1]) if v[0][0] > v[1][0] else (v[1], v[0])
    third = np.cross(h, e)
    m = np.vstack([h, e, third/np.linalg.norm(third)]).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)
