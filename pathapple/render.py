"""Shared drawing for the three workstation layouts in layouts.py: fonts, line icons,
slide labels, the slide overview with the field's footprint, anatomy cards and the
viewport image itself."""
import hashlib
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import PANEL_SIZE, Pose, footprint
from .styles import STYLES

FONTS = {False: ['/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'],
         True: ['/System/Library/Fonts/Supplemental/Arial Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']}
OBJECTIVES = (1.25, 2.5, 5, 10, 20, 40)
SCAN_UM_AT_20X = .494     # GTEx slides were scanned at 20x, about 0.494 µm per pixel


@lru_cache(maxsize=64)
def font(size, bold=False):
    for name in FONTS[bold]:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def hex_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


class Painter:
    def __init__(self, canvas):
        self.c, self.d = canvas, ImageDraw.Draw(canvas)

    def text(self, xy, s, size=12, fill='#222222', anchor='la', bold=False):
        self.d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)

    def width(self, s, size=12, bold=False):
        return self.d.textlength(s, font=font(size, bold))

    def box(self, b, fill=None, outline=None, width=1, radius=0):
        x, y, w, h = b
        if radius:
            self.d.rounded_rectangle((x, y, x+w-1, y+h-1), radius, fill=fill, outline=outline, width=width)
        else:
            self.d.rectangle((x, y, x+w-1, y+h-1), fill=fill, outline=outline, width=width)

    def shade(self, b, colour, alpha):
        """Blend a translucent rectangle over the canvas (floating tool windows)."""
        x, y, w, h = [int(v) for v in b]
        region = np.asarray(self.c.crop((x, y, x+w, y+h))).astype(np.float32)
        tint = np.array(hex_rgb(colour) if isinstance(colour, str) else colour, np.float32)
        self.c.paste(Image.fromarray((region*(1-alpha) + tint*alpha).astype(np.uint8)), (x, y))

    def paste(self, img, xy):
        self.c.paste(img if isinstance(img, Image.Image) else Image.fromarray(img), (int(xy[0]), int(xy[1])))


# ---------------------------------------------------------------- icons
# Material Symbols (Google, Apache 2.0), light weight, for the toolbar glyphs.
SYMBOLS = dict(move='open_with', rect='crop_square', ellipse='circle', line='diagonal_line', polygon='polyline',
               brush='brush', wand='auto_fix_high', points='scatter_plot', contrast='contrast', zoom='zoom_in',
               grid='grid_on', ruler='straighten', gear='settings', pin='location_on', pen='edit',
               expand='open_in_full', info='info', folder='folder', tray='view_agenda', layers='layers',
               chat='chat_bubble', close='close', arrow='arrow_selector_tool', focus='center_focus_strong',
               annotations='pentagon', table='table_rows', eye='visibility', tune='tune', image='image',
               counts='stacked_line_chart', select='select')


@lru_cache(maxsize=1)
def symbol_codes():
    from .inputs import ASSETS
    path = ASSETS / 'fonts' / 'MaterialSymbolsOutlined.codepoints'
    if not path.exists():
        return {}
    return {k: chr(int(v, 16)) for k, v in (line.split() for line in path.read_text().splitlines() if line.strip())}


@lru_cache(maxsize=16)
def symbol_font(size, weight=300):
    from .inputs import ASSETS
    f = ImageFont.truetype(str(ASSETS / 'fonts' / 'MaterialSymbolsOutlined.ttf'), size)
    f.set_variation_by_axes([0, 0, max(20, min(48, size)), weight])
    return f


def icon(P, name, cx, cy, colour, s=8):
    """A toolbar glyph about 2.6s across, centred on (cx, cy): Material Symbols when
    the font is present, else a simple drawn fallback."""
    code = symbol_codes().get(SYMBOLS.get(name, name))
    if code:
        P.d.text((cx, cy), code, font=symbol_font(int(round(s*2.6))), fill=colour, anchor='mm')
        return
    d = P.d
    w = 1
    if name == 'move':
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            d.line((cx, cy, cx+dx*s, cy+dy*s), fill=colour, width=w)
            ax, ay = cx+dx*s, cy+dy*s
            d.polygon([(ax+dx*2, ay+dy*2), (ax-dy*3-dx*2, ay-dx*3-dy*2), (ax+dy*3-dx*2, ay+dx*3-dy*2)], fill=colour)
    elif name == 'rect':
        d.rectangle((cx-s, cy-s*.7, cx+s, cy+s*.7), outline=colour, width=w)
    elif name == 'ellipse':
        d.ellipse((cx-s, cy-s*.7, cx+s, cy+s*.7), outline=colour, width=w)
    elif name == 'line':
        d.line((cx-s, cy+s, cx+s, cy-s), fill=colour, width=w)
    elif name == 'polygon':
        d.polygon([(cx-s, cy+s*.6), (cx-s*.3, cy-s), (cx+s, cy-s*.2), (cx+s*.4, cy+s)], outline=colour)
    elif name == 'brush':
        d.ellipse((cx-s*.5, cy-s, cx+s*.5, cy+s*.2), outline=colour, width=w)
        d.line((cx, cy+s*.2, cx-s*.4, cy+s), fill=colour, width=2)
    elif name == 'wand':
        d.line((cx-s, cy+s, cx+s*.3, cy-s*.3), fill=colour, width=2)
        for a in range(0, 360, 60):
            r = np.deg2rad(a)
            d.line((cx+s*.5, cy-s*.5, cx+s*.5+np.cos(r)*s*.45, cy-s*.5+np.sin(r)*s*.45), fill=colour)
    elif name == 'points':
        for dx, dy in ((-.5, -.4), (.5, -.2), (0, .5)):
            d.ellipse((cx+dx*s-2, cy+dy*s-2, cx+dx*s+2, cy+dy*s+2), outline=colour)
    elif name == 'contrast':
        d.ellipse((cx-s, cy-s, cx+s, cy+s), outline=colour)
        d.pieslice((cx-s, cy-s, cx+s, cy+s), 90, 270, fill=colour)
    elif name == 'zoom':
        d.ellipse((cx-s, cy-s, cx+s*.3, cy+s*.3), outline=colour, width=2)
        d.line((cx+s*.2, cy+s*.2, cx+s, cy+s), fill=colour, width=2)
    elif name == 'grid':
        for k in range(4):
            t = -s + k*2*s/3
            d.line((cx-s, cy+t, cx+s, cy+t), fill=colour)
            d.line((cx+t, cy-s, cx+t, cy+s), fill=colour)
    elif name == 'ruler':
        d.line((cx-s, cy+s*.5, cx+s, cy+s*.5), fill=colour, width=w)
        for k in range(5):
            x = cx-s+k*s/2
            d.line((x, cy+s*.5, x, cy+(s*.5-(5 if k % 2 == 0 else 3))), fill=colour)
    elif name == 'gear':
        d.ellipse((cx-s*.55, cy-s*.55, cx+s*.55, cy+s*.55), outline=colour, width=2)
        for a in range(0, 360, 45):
            r = np.deg2rad(a)
            d.line((cx+np.cos(r)*s*.6, cy+np.sin(r)*s*.6, cx+np.cos(r)*s, cy+np.sin(r)*s), fill=colour, width=2)
    elif name == 'pin':
        d.ellipse((cx-s*.5, cy-s, cx+s*.5, cy), outline=colour, width=w)
        d.line((cx, cy, cx, cy+s), fill=colour, width=w)
    elif name == 'pen':
        d.line((cx-s, cy+s, cx+s*.7, cy-s*.7), fill=colour, width=3)
    elif name == 'expand':
        d.line((cx-s, cy+s, cx+s, cy-s), fill=colour)
        d.line((cx+s, cy-s, cx+s*.2, cy-s), fill=colour)
        d.line((cx+s, cy-s, cx+s, cy-s*.2), fill=colour)
    elif name == 'info':
        d.ellipse((cx-s, cy-s, cx+s, cy+s), outline=colour)
        P.text((cx, cy+1), 'i', int(s*1.4), colour, 'mm', bold=True)
    elif name == 'folder':
        d.rectangle((cx-s, cy-s*.5, cx+s, cy+s*.7), outline=colour)
        d.rectangle((cx-s, cy-s*.8, cx-s*.2, cy-s*.5), outline=colour)
    elif name == 'tray':
        for k in range(3):
            d.rectangle((cx-s, cy-s+k*s*.7, cx+s, cy-s*.55+k*s*.7), outline=colour)
    elif name == 'layers':
        for k in range(3):
            d.polygon([(cx, cy-s+k*4), (cx+s, cy-s*.5+k*4), (cx, cy+k*4), (cx-s, cy-s*.5+k*4)], outline=colour)
    elif name == 'chat':
        d.rounded_rectangle((cx-s, cy-s*.7, cx+s, cy+s*.4), 3, outline=colour)
        d.polygon([(cx-s*.3, cy+s*.4), (cx-s*.6, cy+s), (cx, cy+s*.4)], outline=colour)
    elif name == 'close':
        d.line((cx-s*.6, cy-s*.6, cx+s*.6, cy+s*.6), fill=colour)
        d.line((cx-s*.6, cy+s*.6, cx+s*.6, cy-s*.6), fill=colour)
    elif name == 'arrow':
        d.polygon([(cx-s*.6, cy-s), (cx-s*.6, cy+s*.6), (cx-s*.1, cy+s*.15), (cx+s*.5, cy+s*.6)], outline=colour)
    elif name == 'focus':
        d.ellipse((cx-s, cy-s, cx+s, cy+s), outline=colour)
        d.ellipse((cx-s*.35, cy-s*.35, cx+s*.35, cy+s*.35), fill=colour)


# ---------------------------------------------------------------- data views
def magnification(p):
    """Objective-equivalent magnification of the panel, as viewers report it: 20x is
    one screen pixel per scanned 20x pixel."""
    return 20 * SCAN_UM_AT_20X / (p.width_um / PANEL_SIZE[0])


def nearest_objective(mag):
    return min(OBJECTIVES, key=lambda o: abs(np.log(o/mag)))


def length_label(um):
    return f'{um/1000:g} mm' if um >= 1000 else f'{um:g} µm'


def bar_length(um_per_px, target_px):
    choices = [10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000]
    return min(choices, key=lambda c: abs(np.log(c/(target_px*um_per_px))))


def mmss(t):
    return f'{int(t//60):02}:{t % 60:05.2f}'


@lru_cache(maxsize=96)
def overview(slide_index, box_wh, dim=1.0, glass=None):
    """The slide overview cropped to its tissue and fitted to a box. Returns the image
    and (x0, y0, scale, µm per overview pixel) to map footprints."""
    from .geometry import load_slides
    s = load_slides()[slide_index]
    rgb = s.overview()
    tissue = cv2.resize(s.tissue_mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    bx, by, bw, bh = cv2.boundingRect(tissue)
    pad = round(.05*max(bw, bh))
    x0, y0 = max(0, bx-pad), max(0, by-pad)
    x1, y1 = min(rgb.shape[1], bx+bw+pad), min(rgb.shape[0], by+bh+pad)
    view = rgb[y0:y1, x0:x1].astype(np.float32)
    if glass is not None:
        # Re-map the scanned glass colour to the theme's glass, keeping stain.
        od = -np.log(np.clip(view/s.glass, 1/255, 1))
        view = np.array(glass, np.float32)*np.exp(-od)
    view = np.clip(view*dim, 0, 255).astype(np.uint8)
    W, H = box_wh
    k = min(W/view.shape[1], H/view.shape[0])
    img = Image.fromarray(view).resize((max(1, round(view.shape[1]*k)), max(1, round(view.shape[0]*k))),
                                       Image.Resampling.LANCZOS)
    return img, (x0, y0, k, s.spec['levels']['overview']['um_per_px'])


def footprint_px(slides, si, p, mapping, origin):
    x0, y0, k, um = mapping
    return [(origin[0] + (cx - x0)*k, origin[1] + (cy - y0)*k) for cx, cy in footprint(slides[si], p, um)]


def draw_footprint(P, pts, colour, width=2, halo='#000000', min_ring=0):
    if halo:
        P.d.polygon(pts, outline=halo, width=width+2)
    P.d.polygon(pts, outline=colour, width=width)
    span = max(np.ptp([q[0] for q in pts]), np.ptp([q[1] for q in pts]))
    if span < min_ring:
        cx, cy = np.mean([q[0] for q in pts]), np.mean([q[1] for q in pts])
        r = min_ring*.7
        if halo:
            P.d.ellipse((cx-r-1, cy-r-1, cx+r+1, cy+r+1), outline=halo, width=3)
        P.d.ellipse((cx-r, cy-r, cx+r, cy+r), outline=colour, width=1)


def fit(P, text, width, size, minimum=6, bold=False):
    while size > minimum and P.width(text, size, bold) > width:
        size -= 1
    return size


def slide_label(P, box, slide, light=True):
    """A printed slide label: a 2D code pattern from the specimen ID (decorative, not a
    readable symbol), the stain, the tissue and the specimen ID, sized to fit."""
    x, y, w, h = [int(v) for v in box]
    P.box((x, y, w, h), fill='#fbfbf8' if light else '#d9d6cf', outline='#9a978f')
    n = 9
    side = int(min(w*.42, h*.42))
    cell = max(1, side//n)
    bits = bin(int(hashlib.sha256(slide.spec['specimen'].encode()).hexdigest(), 16))[2:].zfill(256)
    ox, oy = x + 3, y + 3
    for i in range(n):
        for j in range(n):
            if j == 0 or i == n-1 or (i == 0 and j % 2 == 0) or bits[(i*n+j) % 256] == '1':
                P.d.rectangle((ox+j*cell, oy+i*cell, ox+(j+1)*cell-1, oy+(i+1)*cell-1), fill='#1b1b1b')
    inner = w - 6
    tx = ox + n*cell + 3
    P.text((tx, oy), 'H&E', fit(P, 'H&E', x+w-3-tx, max(7, int(h*.16)), bold=True), '#111111', bold=True)
    ty = oy + n*cell + 3
    room = y + h - 3 - ty
    size = max(6, min(int(h*.14), int(room/3.3)))
    words, lines, line = slide.tissue.split(), [], ''
    for wd in words:
        trial = (line + ' ' + wd).strip()
        if P.width(trial, size) > inner and line:
            lines.append(line)
            line = wd
        else:
            line = trial
    lines.append(line)
    lines = lines[:2]
    for k, ln in enumerate(lines):
        P.text((x+3, ty + k*(size+1)), ln, fit(P, ln, inner, size), '#111111')
    sid = slide.spec['specimen'].replace('GTEX-', '')
    P.text((x+3, y+h-3), sid, fit(P, sid, inner, size), '#444444', 'ld')


def glass_slide(P, box, slide_index, slides, label_side='left', light=True, dim=1.0, glass=(236, 236, 238)):
    """A glass slide drawn at 3:1 with its label end and the tissue overview."""
    x, y, w, h = [int(v) for v in box]
    P.box((x, y, w, h), fill=tuple(int(c*dim) for c in glass), outline='#8d8d92' if light else '#55555c', radius=2)
    lw = int(w*.28)
    lx = x if label_side == 'left' else x + w - lw
    slide_label(P, (lx+1, y+1, lw-2, h-2), slides[slide_index], light)
    tx = x + lw + 2 if label_side == 'left' else x + 2
    img, _ = overview(slide_index, (w - lw - 6, h - 4), dim, tuple(int(c*dim) for c in glass))
    P.paste(img, (tx + (w-lw-4-img.width)//2, y + (h-img.height)//2))


@lru_cache(maxsize=256)
def anatomy_card(tissue, size, theme_key, zoom, yaw=0.):
    """A rendered HRA view, cached on disk by its arguments and the anatomy inputs."""
    import json
    from .anatomy import HRA, scene_digest
    key = hashlib.sha256(json.dumps([tissue, size, theme_key, zoom, yaw, scene_digest(), CARD_VERSION]).encode()).hexdigest()[:24]
    path = HRA / 'cards' / f'{key}.npz'
    if path.exists():
        saved = np.load(path, allow_pickle=False)
        return saved['img'], json.loads(str(saved['site'])), str(saved['sex'])
    from .anatomy import view
    img, site, sex = view(tissue, size, dict(theme_key), zoom, yaw)
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, img=img, site=json.dumps(site), sex=sex)
    tmp.replace(path)
    return img, site, sex


CARD_VERSION = 3   # 2: groin removed from the skin shell; 3: rim-only shell


def site_marker(P, site, origin, colour, halo):
    if not site:
        return
    x, y = origin[0] + site['xy'][0], origin[1] + site['xy'][1]
    r = 9
    if site['approx']:
        for a in range(0, 360, 30):
            P.d.arc((x-r, y-r, x+r, y+r), a, a+16, fill=colour, width=2)
    else:
        P.d.ellipse((x-r-1, y-r-1, x+r+1, y+r+1), outline=halo, width=3)
        P.d.ellipse((x-r, y-r, x+r, y+r), outline=colour, width=2)


def viewport(F, size=PANEL_SIZE):
    """The matched field as displayed, in the frame's style."""
    p = F['pose']
    s = F['slides'][p.slide]
    return STYLES[F['style']](s, p, F['match']['level'], *size)


def frame_info(slides, match, original, frame_no, total, time_s, scores, t, style='adaptive', history=()):
    p = Pose(**match['pose'])
    return dict(slides=slides, match=match, pose=p, slide=slides[p.slide], original=original,
                frame_no=frame_no, total=total, time_s=time_s, scores=scores, t=t, style=style,
                history=history, mag=magnification(p), score=match['metrics']['score'])


def spark(P, box, scores, t, line, done, mark, lo=None):
    x, y, w, h = box
    s = np.asarray(scores, float)
    lo = max(0., np.floor(s.min()*10)/10) if lo is None else lo
    pts = [(x + w*j/max(1, len(s)-1), y + h*(1-(v-lo)/max(1e-6, 1-lo))) for j, v in enumerate(s)]
    if len(pts) > 1:
        P.d.line(pts, fill=line, width=1)
        if t:
            P.d.line(pts[:t+1], fill=done, width=1)
    P.d.line((pts[t][0], y-2, pts[t][0], y+h+2), fill=mark)
    P.d.ellipse((pts[t][0]-2, pts[t][1]-2, pts[t][0]+2, pts[t][1]+2), fill=mark)
