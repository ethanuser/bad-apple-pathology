"""Contrast fitting and silhouette metrics. White and black classes receive equal weight."""
import cv2
import numpy as np


def down(image, width):
    return cv2.resize(image, (width, width * 3 // 4), interpolation=cv2.INTER_AREA)


def unit_rows(a):
    a = np.asarray(a, dtype=np.float32)
    centered = a - a.mean(axis=1, keepdims=True)
    return centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-6)


def regional_features(images, query=False):
    """Equalize contrast in a 4 × 3 grid, so small details can influence retrieval."""
    n, h, w = images.shape
    tiles = images.reshape(n, 3, h//3, 4, w//4).transpose(0, 1, 3, 2, 4).reshape(n, 12, -1)
    tiles = tiles - tiles.mean(axis=2, keepdims=True)
    norms = np.linalg.norm(tiles, axis=2, keepdims=True)
    features = tiles / np.maximum(norms, .025*np.sqrt(tiles.shape[2]))
    if query:
        # Only target regions with an observable silhouette boundary carry weight.
        weights = np.sqrt(norms)
        features *= weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    return np.ascontiguousarray(features.reshape(n, -1), dtype=np.float32)


def ncc(a, b):
    x = a.astype(np.float32).ravel()
    y = b.astype(np.float32).ravel()
    x -= x.mean()
    y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < 1e-6:
        return float(1 - np.abs(a.astype(float) - b).mean() / 255)
    return float(np.clip(np.dot(x, y) / denom, -1, 1))


def mask_iou(pred, target):
    intersection = np.count_nonzero(pred & target)
    union = np.count_nonzero(pred | target)
    n = pred.size
    white = intersection / union if union else 1.0
    black = (n - union) / (n - intersection) if n != intersection else 1.0
    return (white + black) / 2


def edges(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    e = np.zeros_like(mask)
    e[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    e[1:, :] |= mask[1:, :] != mask[:-1, :]
    return e


def boundary_f1(a, b):
    ea, eb = edges(a), edges(b)
    if not ea.any() or not eb.any():
        return float(ea.any() == eb.any())
    def hit(source, dest):
        distance = cv2.distanceTransform(1 - dest, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return np.mean(distance[source != 0] <= 2.0)
    p, r = hit(ea, eb), hit(eb, ea)
    return float(2*p*r / (p+r)) if p+r else 0.


WEIGHTS = dict(miou=.28, boundary_f1=.27, detail=.20, ncc=.125, regional_ncc=.125)


def thin_parts(truth, size=5):
    """Target pixels in features under about five pixels wide at 96 × 72: hair strands,
    fingers, narrow gaps. Opening removes them, so they are the difference."""
    t = np.asarray(truth, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    white = t.astype(bool) & ~cv2.morphologyEx(t, cv2.MORPH_OPEN, kernel).astype(bool)
    black = ~t.astype(bool) & ~cv2.morphologyEx(1-t, cv2.MORPH_OPEN, kernel).astype(bool)
    return white, black


def detail_recall(pred, truth, thin=None):
    """Mean fraction of each class's thin target features that the prediction reproduces.
    Without thin features the term defers to balanced IoU."""
    white, black = thin_parts(truth) if thin is None else thin
    parts = [pred[white].mean()] if white.any() else []
    parts += [(~pred[black]).mean()] if black.any() else []
    return float(np.mean(parts)) if parts else mask_iou(pred, truth)


def resolution_bonus(width_mm, pitch):
    """Selection-only preference for crops with more native voxels across the panel.
    Zero at 32 voxels, rising with log size to RES_WEIGHT at 128 voxels."""
    voxels = np.asarray(width_mm) / float(np.sqrt(pitch[0]*pitch[1]))
    return RES_WEIGHT*np.clip(np.log2(voxels/32)/2, 0, 1)


RES_WEIGHT = .025
# Thresholds this close to zero trace the brain-extraction mask's voxel staircase
# rather than tissue contrast. They remain allowed at a small selection cost.
MASK_EDGE_LEVEL, MASK_EDGE_PENALTY = 20, .025


def score_image(image, target):
    pred, truth = image >= 128, target >= 128
    iou = mask_iou(pred, truth)
    edge = boundary_f1(pred, truth)
    detail = detail_recall(pred, truth)
    corr = ncc(image, target)
    features = regional_features(image[None].astype(np.float32)/255)
    weights = regional_features(target[None].astype(np.float32)/255, query=True)
    local = float(np.clip((features*weights).sum(), -1, 1)) if weights.any() else corr
    metrics = dict(miou=iou, boundary_f1=edge, detail=detail, ncc=corr, regional_ncc=local)
    metrics['score'] = sum(w*max(0., metrics[k]) for k, w in WEIGHTS.items())
    return metrics


def fit_levels(patches, target, invert):
    """Pick each patch's threshold maximizing balanced IoU from 64-bin class histograms."""
    n = len(patches)
    truth = (target >= 128).ravel()
    # Work in display polarity. With level k-0.5, >= k maps to displayed white.
    adjusted = np.where(np.asarray(invert)[:, None, None], 255-patches, patches)
    bins = (adjusted.reshape(n, -1) // 4).astype(np.int32) + 64*np.arange(n, dtype=np.int32)[:, None]
    def counts(values):
        hist = np.bincount(values.ravel(), minlength=n*64).reshape(n, 64)
        return np.concatenate([hist[:, ::-1].cumsum(axis=1)[:, ::-1], np.zeros((n, 1))], axis=1)
    tp = counts(bins[:, truth])
    fp = counts(bins[:, ~truth])
    pos, neg = truth.sum(), (~truth).sum()
    def ratio(a, b):
        return np.divide(a, b, out=np.ones_like(a, dtype=float), where=b != 0)
    iou = .5*(ratio(tp, pos + fp) + ratio(neg-fp, neg+pos-tp))
    k = iou.argmax(axis=1)
    level = 4*k - .5
    return np.where(invert, 255-level, level), iou[np.arange(n), k]
