"""Coarse retrieval on the GPU over slides x stain channels x magnification bands,
then local refinement of position, zoom, angle and silhouette smoothing on the CPU."""
from dataclasses import replace
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .geometry import ASPECT, BASE_UM, Pose, field, inside, values
from .metrics import down, fit_levels, ncc, regional_features, score_image, thin_parts

COARSE_W, REFINE_W, METRIC_W = 32, 48, 96
KEEP, CHUNK = 48, 1024
# Width bands (µm). Each band samples a pyramid level matched to its thumbnail pitch.
BANDS = ((700., 1500.), (1500., 3200.), (3200., 6800.), (6800., 14000.))
SMOOTH_START, SMOOTH_RANGE = .012, (.002, .045)
# Local refinement: (perturbation scale, neighbours per seed, seeds kept) per round. The
# MRI/CT edition found six progressively finer rounds the one cheap gain (+0.02).
ROUNDS = ((1.15, 128, 32), (.60, 128, 24), (.30, 96, 16), (.15, 64, 10), (.07, 48, 8), (.035, 32, 7))
ROUNDS_FAST = ((1.15, 16, 10), (.60, 24, 7), (.28, 20, 7))
# Selection-only cost for crops that are mostly empty glass.
GLASS_WEIGHT = .08


def pick_device(name='auto'):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def unit_rows(a):
    a = a - a.mean(dim=1, keepdim=True)
    return a / a.norm(dim=1, keepdim=True).clamp_min(1e-6)


def regional(images, query=False):
    n, h, w = images.shape
    tiles = images.reshape(n, 3, h//3, 4, w//4).permute(0, 1, 3, 2, 4).reshape(n, 12, -1)
    tiles = tiles - tiles.mean(dim=2, keepdim=True)
    norms = tiles.norm(dim=2, keepdim=True)
    features = tiles / norms.clamp_min(.025*np.sqrt(tiles.shape[2]))
    if query:
        weights = norms.sqrt()
        features = features * weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return features.reshape(n, -1)


def gradient_query(imgs):
    dx, dy = imgs.diff(dim=2), imgs.diff(dim=1)
    lap = torch.zeros_like(imgs)
    lap[:, :, 1:] += dx
    lap[:, :, :-1] -= dx
    lap[:, 1:, :] += dy
    lap[:, :-1, :] -= dy
    norm = ((dx*dx).sum(dim=(1, 2)) + (dy*dy).sum(dim=(1, 2))).sqrt()
    return (lap.reshape(len(imgs), -1) / norm.clamp_min(1e-6)[:, None]).T.contiguous()


def gradient_norm(x, h, w):
    img = x.reshape(-1, h, w)
    dx, dy = img.diff(dim=2), img.diff(dim=1)
    return ((dx*dx).sum(dim=(1, 2)) + (dy*dy).sum(dim=(1, 2))).sqrt().clamp_min(1e-3)


def propose(slide, band, rng, count):
    ys, xs = np.nonzero(slide.tissue_mask)
    if not len(xs):
        return np.empty((0, 5), np.float32)
    picks = rng.integers(len(xs), size=count)
    width = np.exp(rng.uniform(*np.log(band), count))
    x = (xs[picks] + .5)*slide.base_um*4 + rng.uniform(-.4, .4, count)*width
    y = (ys[picks] + .5)*slide.base_um*4 + rng.uniform(-.4, .4, count)*width
    params = np.column_stack([x, y, width, rng.uniform(-180, 180, count), np.zeros(count)]).astype(np.float32)
    return params[inside(params, slide.size_um)]


class Pools:
    def __init__(self, targets, device):
        self.value = torch.full((targets, KEEP), -np.inf, device=device)
        self.payload = torch.zeros((targets, KEEP, 9), device=device)

    def update(self, quality, signed, flip, payload):
        m, t = quality.shape
        k = min(KEEP, m)
        top, at = quality.topk(k, dim=0)                                       # (k, T)
        rows = payload[at]                                                     # (k, T, 7)
        extra = torch.stack([flip.gather(0, at).float(), (signed.gather(0, at) < 0).float()], dim=-1)
        cand = torch.cat([rows, extra], dim=-1).permute(1, 0, 2)               # (T, k, 9)
        value = torch.cat([self.value, top.T], dim=1)
        best, idx = value.topk(KEEP, dim=1)
        self.payload = torch.cat([self.payload, cand], dim=1).gather(1, idx[..., None].expand(t, KEEP, 9))
        self.value = best

    def ranked(self):
        out = []
        for v, rows in zip(self.value.cpu().numpy(), self.payload.cpu().numpy()):
            out.append([Pose(int(r[5]), int(r[6]), float(r[0]), float(r[1]), float(r[2]), float(r[3]),
                             bool(r[7]), bool(r[8]), SMOOTH_START) for val, r in zip(v, rows) if np.isfinite(val)])
        return out


def retrieve(slides, targets, seed, samples, width=COARSE_W, device='auto'):
    dev = pick_device(device)
    rng = np.random.default_rng(seed)
    h = width*3//4
    images = np.array([down(f, width) for f in targets])
    n = len(images)
    bank = torch.from_numpy(np.concatenate([images, images[:, :, ::-1]]).astype(np.float32)).to(dev)
    query = unit_rows(bank.reshape(2*n, -1)).T.contiguous()
    sigma = 1.35*width/16
    highpass = torch.from_numpy(np.array([v-cv2.GaussianBlur(v, (0, 0), sigma) for v in bank.cpu().numpy()])).to(dev)
    contour_query = (.62*query + .38*unit_rows(highpass.reshape(2*n, -1)).T).contiguous()
    shape_query = torch.cat([contour_query, gradient_query(bank)], dim=1)
    regional_query = regional(bank/255, query=True).reshape(2*n, 3, 4, h//3, width//4) \
        .permute(0, 1, 3, 2, 4).reshape(2*n, -1).T.contiguous()
    tiles = F.one_hot(((torch.arange(h, device=dev)[:, None]//(h//3))*4 + torch.arange(width, device=dev)[None]//(width//4)).reshape(-1), 12).float()
    per_tile = h*width/12
    gx = ((torch.arange(width, device=dev, dtype=torch.float32)+.5)/width-.5)[None, None, :]
    gy = ((torch.arange(h, device=dev, dtype=torch.float32)+.5)/h-.5)[None, :, None]/ASPECT
    pools, shape_pools = Pools(n, dev), Pools(n, dev)
    views = 0
    begin = time.perf_counter()

    def norms(x, tol):
        d = x.shape[1]
        s1, s2 = x.sum(1), (x*x).sum(1)
        return (s2 - s1*s1/d).clamp_min(0).sqrt().clamp_min(tol*np.sqrt(d))

    def score(plane, pitch, params, levels, meta):
        P = torch.from_numpy(params).to(dev)
        x, y, wd, ang, mir = (P[:, k][:, None, None] for k in range(5))
        a = torch.deg2rad(ang)
        c, s = wd*torch.cos(a), wd*torch.sin(a)
        u = gx*(1-2*mir)
        px = (x + c*u - s*gy) / pitch
        py = (y + s*u + c*gy) / pitch
        H, W = plane.shape
        grid = torch.stack([2*px/W-1, 2*py/H-1], dim=-1)[None].reshape(1, -1, width, 2)
        flat = F.grid_sample(plane[None, None], grid, mode='bilinear', padding_mode='border',
                             align_corners=False).reshape(len(params), h*width)
        corr = (flat @ query) / norms(flat, .01)[:, None]
        lv = torch.from_numpy(levels).to(dev)[:, None]
        best = torch.zeros_like(corr)
        for center in (lv, 255-lv):
            soft = ((flat - center)/55).clamp(-.5, .5)
            both = soft @ shape_query
            cc = both[:, :2*n] / norms(soft, 1e-3)[:, None]
            grad = both[:, 2*n:] / gradient_norm(soft, h, width)[:, None]
            t1, t2 = soft @ tiles, (soft*soft) @ tiles
            inv = 1/(t2 - t1*t1/per_tile).clamp_min(0).sqrt().clamp_min(.025*np.sqrt(per_tile))
            local = (soft * inv[:, tiles.argmax(1)]) @ regional_query
            cc = .32*cc + .23*grad + .45*local
            best = torch.where(cc.abs() > best.abs(), cc, best)
        payload = torch.cat([P, torch.from_numpy(meta).to(dev)], dim=1)
        for scores, pool in ((corr, pools), (best, shape_pools)):
            direct, reflected = scores[:, :n], scores[:, n:]
            flip = reflected.abs() > direct.abs()
            signed = torch.where(flip, reflected, direct)
            pool.update(signed.abs(), signed, flip, payload)

    for si, slide in enumerate(slides):
        for band in BANDS:
            proposals = propose(slide, band, rng, samples)
            views += 3*len(proposals)
            # Pyramid level at about half the thumbnail pitch of the narrowest crop.
            k, pitch = slide.level_for(band[0]/width/2)
            for channel in range(3):
                plane = torch.from_numpy(cv2.GaussianBlur(slide.pyramid[channel][k].astype(np.float32), (0, 0), .8)).to(dev)
                for start in range(0, len(proposals), 8*CHUNK):
                    p = proposals[start:start+8*CHUNK]
                    lv = rng.uniform(60, 235, size=len(p)).astype(np.float32)
                    meta = np.tile(np.array([[si, channel]], np.float32), (len(p), 1))
                    score(plane, pitch, p, lv, meta)
        print(f'Coarse search {si+1}/{len(slides)} slides, {views:,} crops', flush=True)
    return pools.ranked(), shape_pools.ranked(), dict(coarse_seconds=time.perf_counter()-begin,
                                                     crops=views, device=str(dev), coarse_width=width)


def tissue_share(slide, p, w=24):
    """Share of the crop on stained tissue, from the 16 µm tissue mask."""
    from .geometry import sample
    m = sample(slide.tissue_mask.astype(np.uint8)*255, 4*slide.base_um, [values(p)], w, w*3//4, border=0)[0]
    return float((m > 127).mean())


def prior(slide, p):
    return GLASS_WEIGHT*float(np.clip((.3 - tissue_share(slide, p))/.3, 0, 1))


def neighbors(p, slide, rng, count, scale):
    delta = rng.normal(size=(count, 5)) * scale
    rows = []
    for dx, dy, dw, da, ds in delta:
        rows.append(replace(p, x_um=p.x_um+dx*p.width_um*.09, y_um=p.y_um+dy*p.width_um*.09,
                            width_um=float(np.clip(p.width_um*np.exp(dw*.13), *slide.width_clip)),
                            angle_deg=float((p.angle_deg+da*11+180) % 360-180),
                            smooth=float(np.clip(p.smooth*np.exp(ds*.45), *SMOOTH_RANGE))))
    ok = inside([values(q) for q in rows], slide.size_um)
    return [q for q, keep in zip(rows, ok) if keep]


def choose_seeds(poses, quality, n):
    result = []
    for i in np.argsort(-quality, kind='stable'):
        p = poses[i]
        if any(p.slide == q.slide and p.channel == q.channel and p.mirror == q.mirror and p.invert == q.invert
               and abs(p.x_um-q.x_um)+abs(p.y_um-q.y_um) < .06*p.width_um
               and abs(p.width_um-q.width_um) < .07*p.width_um
               and abs((p.angle_deg-q.angle_deg+180) % 360-180) < 7 for q in result):
            continue
        result.append(p)
        if len(result) == n:
            break
    return result


def gradient_np(images):
    imgs = np.asarray(images, dtype=np.float32)
    dx, dy = np.diff(imgs, axis=2), np.diff(imgs, axis=1)
    lap = np.zeros_like(imgs)
    lap[:, :, 1:] += dx
    lap[:, :, :-1] -= dx
    lap[:, 1:, :] += dy
    lap[:, :-1, :] -= dy
    norm = np.sqrt((dx*dx).sum(axis=(1, 2)) + (dy*dy).sum(axis=(1, 2)))
    return (lap.reshape(len(imgs), -1) / np.maximum(norm[:, None], 1e-6)).T


def refine(slides, poses, target, rng):
    low = down(target, REFINE_W)
    white, black = thin_parts(low >= 128, 3)
    g = gradient_np(low[None])
    q = regional_features(low[None].astype(np.float32)/255, query=True)
    for scale, n, k in ROUNDS:
        cands = poses + [c for p in poses for c in neighbors(p, slides[p.slide], rng, n, scale)]
        small = np.stack([field(slides[c.slide], c, REFINE_W, REFINE_W*3//4) for c in cands])
        small = np.clip(np.rint(small), 0, 255).astype(np.uint8)
        inv = np.array([c.invert for c in cands])
        levels, iou = fit_levels(small, low, inv)
        soft = np.clip((small.astype(np.float32) - levels[:, None, None])/62 + .5, 0, 1)
        soft = np.ascontiguousarray(np.where(inv[:, None, None], 1-soft, soft), dtype=np.float32)
        agreement = 1 - np.abs(soft - low.astype(np.float32)/255).mean(axis=(1, 2))
        dx, dy = np.diff(soft, axis=2), np.diff(soft, axis=1)
        gn = np.maximum(np.sqrt((dx*dx).sum(axis=(1, 2)) + (dy*dy).sum(axis=(1, 2))), 1e-3)
        with np.errstate(all='ignore'):
            edge = (soft.reshape(len(soft), -1) @ g).ravel() / gn
        local = (regional_features(soft)*q).sum(axis=1)
        pred = soft.reshape(len(soft), -1) >= .5
        parts = [pred[:, white.ravel()].mean(axis=1)] if white.any() else []
        parts += [(~pred[:, black.ravel()]).mean(axis=1)] if black.any() else []
        detail = np.mean(parts, axis=0) if parts else iou
        quality = .35*iou + .08*agreement + .17*edge + .20*local + .20*detail
        quality -= np.array([prior(slides[c.slide], c) for c in cands])
        poses = choose_seeds(cands, quality, k)
    return poses


def evaluate(slides, p, target):
    from .geometry import silhouette
    s = slides[p.slide]
    raw = np.clip(np.rint(field(s, p, METRIC_W*2, METRIC_W*3//2)), 0, 255).astype(np.uint8)
    level, _ = fit_levels(raw[None], cv2.resize(target, (raw.shape[1], raw.shape[0])), [p.invert])
    best = None
    for offset in (-8, -4, 0, 4, 8):
        lv = float(np.clip(level[0] + offset, .5, 254.5))
        image = silhouette(s, p, lv, METRIC_W, METRIC_W*3//4)
        metrics = score_image(image, target)
        if best is None or metrics['score'] > best['metrics']['score']:
            best, best_image = dict(pose=p.record(), level=lv, metrics=metrics), image
    # Reported only, for comparison with ikuto32's Earth/Moon edition: NCC at 32 x 24.
    # (Their Earth metric, mean foreground/background IoU at 128, is 'miou' here.)
    best['metrics']['ncc_32x24'] = ncc(down(best_image, 32), down(target, 32))
    best['objective'] = best['metrics']['score'] - prior(s, p)
    return best


def follow(p, previous, current, slide):
    """Translate and uniformly zoom the same field using silhouette moments, so a
    moving figure can keep its tissue between frames."""
    a, b = previous >= 128, current >= 128
    if a.mean() > .5:
        a, b = ~a, ~b
    if not a.any() or not b.any() or a.all() or b.all():
        return p
    ratio = np.sqrt(b.sum()/a.sum())
    if not .64 <= ratio <= 1.56:
        return p
    h, w = a.shape

    def centroid(mask):
        y, x = np.nonzero(mask)
        return np.array([(x.mean()+.5-w/2)/w, (y.mean()+.5-h/2)/w])
    width = float(np.clip(p.width_um/ratio, *slide.width_clip))
    u, v = p.width_um*centroid(a) - width*centroid(b)
    if p.mirror:
        u = -u
    ang = np.deg2rad(p.angle_deg)
    q = replace(p, width_um=width, x_um=p.x_um + np.cos(ang)*u - np.sin(ang)*v,
                y_um=p.y_um + np.sin(ang)*u + np.cos(ang)*v)
    return q if inside([values(q)], slide.size_um)[0] else p


CUT_COST = .038


def switch_cost(a, b):
    if (a.slide, a.channel, a.mirror, a.invert) != (b.slide, b.channel, b.mirror, b.invert):
        return CUT_COST
    distance = np.hypot(a.x_um-b.x_um, a.y_um-b.y_um)/a.width_um
    turn = abs((a.angle_deg-b.angle_deg+180) % 360-180)/90
    return min(CUT_COST, .018*(distance + turn + abs(np.log(b.width_um/a.width_um))
                               + .5*abs(np.log(b.smooth/a.smooth))))


def sequence(choices):
    """Maximum-score path through each frame's candidates, charging a small cost for
    discontinuous cuts."""
    prior_ = np.zeros(1)
    links, old = [], []
    for i, row in enumerate(choices):
        poses = [Pose(**c['pose']) for c in row]
        if i:
            costs = np.array([[switch_cost(a, b) for b in poses] for a in old])
            path = prior_[:, None] - costs
            parent = path.argmax(axis=0)
            prior_ = path[parent, np.arange(len(row))]
            links.append(parent)
        prior_ = prior_ + np.array([c['objective'] for c in row])
        old = poses
    index = int(prior_.argmax())
    result = []
    for i in range(len(choices)-1, -1, -1):
        result.append(choices[i][index])
        if i:
            index = int(links[i-1][index])
    return result[::-1]


def load_retrieval(path, key):
    import gzip
    import json
    if path is None or not path.exists():
        return None
    with gzip.open(path, 'rt') as f:
        saved = json.load(f)
    if saved['key'] != key:
        print(f'Ignoring {path.name}: it was made with different settings or inputs', flush=True)
        return None
    print(f'Reusing retrieval from {path.name}', flush=True)
    poses = [[[Pose(**p) for p in pool] for pool in group] for group in saved['pools']]
    return poses[0], poses[1], saved['stats']


def save_retrieval(path, key, pools, shape_pools, stats):
    import gzip
    import json
    temp = path.with_suffix('.tmp')
    with gzip.open(temp, 'wt') as f:
        json.dump(dict(key=key, stats=stats, pools=[[[p.record() for p in pool] for pool in group]
                                                   for group in (pools, shape_pools)]), f)
    temp.replace(path)


def default_workers():
    import os
    for name in ('SLURM_CPUS_PER_TASK', 'CPU_LIMIT'):
        if os.environ.get(name):
            return max(1, int(float(os.environ[name])))
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


_BLOCK_JOB = {}


def _run_block(bounds):
    return _BLOCK_JOB['fn'](*bounds)


MPS_USED = False


def run_blocks(fn, bounds, workers):
    """Forked processes on Linux, or on a Mac whose GPU this process has not used
    (Metal is not fork-safe); threads otherwise."""
    import multiprocessing
    import sys
    from concurrent.futures import ThreadPoolExecutor
    if workers > 1 and (sys.platform == 'linux' or not MPS_USED):
        _BLOCK_JOB['fn'] = fn
        try:
            with multiprocessing.get_context('fork').Pool(workers) as pool:
                return pool.map(_run_block, bounds, chunksize=1), 'processes'
        finally:
            _BLOCK_JOB.clear()
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda b: fn(*b), bounds)), 'threads'


def pipeline(slides, targets, seed, samples, coarse_width=COARSE_W, device='auto', cache=None, cache_key=None,
             workers=None):
    """Coarse retrieval (saved, so an interrupted refinement resumes without it), then
    per-frame refinement seeded by the previous frame, reverse propagation and a joint
    choice of the sequence. Returns the grayscale-NCC baseline, the selection and stats."""
    global MPS_USED
    saved = load_retrieval(cache, cache_key)
    if saved:
        pools, shape_pools, stats = saved
    else:
        pools, shape_pools, stats = retrieve(slides, targets, seed, samples, coarse_width, device)
        MPS_USED = MPS_USED or stats['device'] == 'mps'
        if cache is not None:
            save_retrieval(cache, cache_key, pools, shape_pools, stats)
    metric_targets = [down(f, METRIC_W) for f in targets]
    begin = time.perf_counter()
    baseline = [evaluate(slides, pool[0], target) for pool, target in zip(pools, metric_targets)]
    stats['baseline_finish_seconds'] = time.perf_counter() - begin
    begin = time.perf_counter()

    def refine_block(first, last):
        picked, rows = None, []
        for t in range(first, last):
            target = metric_targets[t]
            seeds = list(dict.fromkeys(pools[t][:24] + shape_pools[t]))
            if picked:
                prev = Pose(**picked['pose'])
                seeds += [prev, follow(prev, metric_targets[t-1], target, slides[prev.slide])]
            finalists = refine(slides, seeds, target, np.random.default_rng(seed+10000+t))
            choices = [baseline[t]] + [evaluate(slides, p, target) for p in finalists]
            if picked:
                choices.append(evaluate(slides, Pose(**picked['pose']), target))
            picked = max(choices, key=lambda c: c['objective'])
            rows.append(choices)
        print(f'Refined frames {first+1}-{last} of {len(targets)}', flush=True)
        return rows

    workers = max(1, min(workers or default_workers(), len(targets)//8 or 1))
    edges = np.linspace(0, len(targets), workers+1).round().astype(int)
    blocks, mode = run_blocks(refine_block, list(zip(edges[:-1].tolist(), edges[1:].tolist())), workers)
    rows = [r for block in blocks for r in block]
    stats['refine_workers'], stats['refine_parallelism'] = workers, mode
    for t in range(len(rows)-2, -1, -1):
        following = max(rows[t+1], key=lambda c: c['objective'])
        p = Pose(**following['pose'])
        rows[t].append(evaluate(slides, follow(p, metric_targets[t+1], metric_targets[t], slides[p.slide]),
                                metric_targets[t]))
    selected = sequence(rows)
    stats['shape_finish_seconds'] = time.perf_counter() - begin
    return baseline, selected, stats
