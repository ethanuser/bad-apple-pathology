"""Three workstation layouts, each modelled on the conventions of a real slide viewer.

A  Analysis workstation   after QuPath: menu and icon toolbar with a magnification
                          readout, an Image tab of name and value rows (including the
                          slide's own stain vectors), the overview inset with the field
                          box, a grey scale bar and a cursor readout.
B  Clinical slide tray    after Aperio WebViewer DX: the case's slides in a two-column
                          tray, a vertical tool strip, the slide label, thumbnail with a
                          heat map of where the video has looked, the colour-coded zoom
                          control, rotation dial, magnifier and scale bar.
C  Scanner viewer         after Hamamatsu NDP.view2: the field full-bleed with dark
                          floating tool windows (browse, slide image, lens, map).

Every layout carries the source frame and the Human Reference Atlas body with GTEx's
registered sampling site."""
import numpy as np
from PIL import Image

from .geometry import CHANNELS, PANEL_SIZE
from .render import (OBJECTIVES, Painter, anatomy_card, bar_length, draw_footprint, footprint_px,
                     glass_slide, hex_rgb, icon, length_label, mmss, nearest_objective, overview,
                     site_marker, slide_label, spark, viewport)

SIZE = (1440, 1080)


def theme_key(**kw):
    return tuple(sorted(kw.items()))


def stain_text(v):
    return ' '.join(f'{x:.3f}' for x in v)


def site_caption(F):
    from .anatomy import TISSUES
    spec = TISSUES.get(F['slide'].tissue, {})
    if spec.get('block') or spec.get('uberon'):
        return 'GTEx registered block'
    return 'Approximate site' if spec.get('approx') else 'HRA structure'


# =====================================================================  A
A = dict(chrome='#ececec', panel='#ffffff', line='#c4c4c4', ink='#1f1f1f', mute='#6f6f6f', red='#d9302c',
         blue='#2f6fd0', tab='#f7f7f7')
A_ANAT = theme_key(map_bg=(250, 250, 250), map_organ=(196, 178, 186), map_line=(120, 120, 130),
                   accent_rgb=(217, 48, 44), block_rgb=(40, 40, 40))


def workstation(F):
    canvas = Image.new('RGB', SIZE, A['chrome'])
    P = Painter(canvas)
    p, s, slides = F['pose'], F['slide'], F['slides']
    W, H = SIZE
    # Title bar, menu bar, toolbar.
    P.box((0, 0, W, 28), fill='#ffffff')
    P.d.line((0, 27, W, 27), fill=A['line'])
    icon(P, 'zoom', 17, 14, '#2b2b2b', 6)
    P.text((32, 14), f"Bad Apple!!  {s.spec['specimen']}.dcm", 13, A['ink'], 'lm')
    P.d.line((W-116, 14, W-104, 14), fill=A['ink'])
    P.d.rectangle((W-76, 8, W-64, 20), outline=A['ink'])
    icon(P, 'close', W-30, 14, A['ink'], 6)
    P.box((0, 28, W, 24), fill='#ffffff')
    x = 14
    for m in ('File', 'Edit', 'Tools', 'View', 'Objects', 'Stain', 'Measure', 'Analyze', 'Atlas', 'Help'):
        P.text((x, 40), m, 13, A['ink'], 'lm')
        x += P.width(m, 13) + 20
    P.d.line((0, 52, W, 52), fill=A['line'])
    ty, x = 70, 10

    def button(name=None, on=False, red=False, label=None, width=30):
        nonlocal x
        P.box((x, ty-13, width, 26), fill='#d9d9d9' if on else '#f7f7f7', outline='#b8b8b8', radius=3)
        colour = '#c62828' if red else '#333333'
        if label:
            P.text((x+width/2, ty), label, 13, colour, 'mm', bold=red)
        else:
            icon(P, name, x+width/2, ty, colour, 7)
        x += width + 4

    def gap():
        nonlocal x
        x += 8
    button('ruler')
    gap()
    button('move', on=True)
    for name in ('rect', 'ellipse', 'line', 'polygon', 'brush', 'wand', 'points'):
        button(name, red=True)
    gap()
    button(label='S')
    button('contrast')
    P.text((x+8, ty), f"{F['mag']:.2f}x", 13, A['ink'], 'lm')
    x += 66
    button('zoom')
    gap()
    button('annotations', red=True, on=True)
    button('points', red=True, on=True)
    button('layers')
    P.d.line((x+8, ty, x+128, ty), fill='#a6a6a6', width=2)
    P.d.ellipse((x+110, ty-6, x+122, ty+6), fill='#f7f7f7', outline='#8a8a8a')
    x += 144
    button('grid')
    button(label='x,y')
    button('ruler')
    button('table')
    gap()
    button('gear')
    P.d.line((0, 88, W, 88), fill=A['line'])

    # Left pane: tabs and the Image table.
    lx, lw = 6, 388
    x = lx
    for t in ('Project', 'Image', 'Annotations', 'Hierarchy', 'Workflow'):
        tw = P.width(t, 12) + 14
        on = t == 'Image'
        P.box((x, 94, tw, 22), fill='#ffffff' if on else A['tab'], outline=A['line'])
        if on:
            P.d.line((x+1, 115, x+tw-2, 115), fill='#ffffff')
        P.text((x+tw/2, 105), t, 12, A['ink'], 'mm')
        x += tw - 1
    fine = s.spec['levels']['fine']
    rows = [('Name', f"{s.spec['specimen']}.dcm"), ('Tissue', s.tissue), ('Image type', 'Brightfield (H&E)'),
            ('Pixel size', f"{fine['um_per_px']:.4f} µm"),
            ('Dimensions', f"{fine['width']} × {fine['height']} px"),
            ('Field', f'{p.width_um/1000:.2f} × {p.width_um*.75/1000:.2f} mm'),
            ('Rotation', f'{p.angle_deg:+.1f}°' + ('  mirrored' if p.mirror else '')),
            ('Channel', CHANNELS[p.channel].capitalize()), ('Threshold', f"{F['match']['level']:.1f}"),
            ('Edge smoothing', f'{p.smooth*p.width_um:.0f} µm'),
            ('Stain 1', 'Hematoxylin: ' + stain_text(s.stains[0])), ('Stain 2', 'Eosin: ' + stain_text(s.stains[1]))]
    top = 116
    P.box((lx, top, lw, 22*(len(rows)+1)+1), fill='#ffffff', outline=A['line'])
    P.box((lx, top, lw, 22), fill='#f2f2f2', outline=A['line'])
    split = lx + 140
    P.text(((lx+split)/2, top+11), 'Name', 12, A['ink'], 'mm', bold=True)
    P.text(((split+lx+lw)/2, top+11), 'Value', 12, A['ink'], 'mm', bold=True)
    y = top + 22
    for name, value in rows:
        P.d.line((lx, y+21, lx+lw-1, y+21), fill='#e6e6e6')
        P.text((lx+6, y+11), name, 12, A['ink'], 'lm')
        colour = {'Stain 1': '#3f4fb0', 'Stain 2': '#c8327a'}.get(name, A['ink'])
        P.text((split+6, y+11), value, 12, colour, 'lm')
        y += 22
    P.d.line((split, top, split, y), fill='#e6e6e6')

    # Lower pane: associated images, as QuPath lists macro, thumbnail and label.
    ay = y + 7
    P.box((lx, ay, lw, H-ay-8), fill='#ffffff', outline=A['line'])
    iw = lw - 16
    P.text((lx+8, ay+12), 'Reference', 12, A['ink'], 'lm', bold=True)
    P.text((lx+lw-8, ay+12), f"Frame {F['frame_no']}  {mmss(F['time_s'])}", 11, A['mute'], 'rm')
    ref_h = round(iw*3/4)
    ref = Image.fromarray(F['original']).convert('RGB').resize((iw, ref_h), Image.Resampling.LANCZOS)
    P.paste(ref, (lx+8, ay+24))
    P.box((lx+8, ay+24, iw, ref_h), outline=A['line'])
    at = ay + 24 + ref_h + 8
    P.text((lx+8, at+12), 'Atlas', 12, A['ink'], 'lm', bold=True)
    P.text((lx+lw-8, at+12), s.tissue, 11, A['mute'], 'rm')
    top2, bottom = at + 24, H - 8 - 22
    bw0 = 118
    img, site, sex = anatomy_card(s.tissue, (bw0, bottom-top2), A_ANAT, 'body')
    P.paste(img, (lx+8, top2))
    site_marker(P, site, (lx+8, top2), A['red'], '#ffffff')
    ox = lx + 8 + bw0 + 6
    img, site, _ = anatomy_card(s.tissue, (lx+lw-8-ox, bottom-top2), A_ANAT, 'organ', 25.)
    P.paste(img, (ox, top2))
    site_marker(P, site, (ox, top2), A['red'], '#ffffff')
    P.box((lx+8, top2, iw, bottom-top2), outline=A['line'])
    P.d.line((ox-3, top2, ox-3, bottom), fill=A['line'])
    P.text((lx+8, H-19), site_caption(F), 11, A['mute'], 'lm')
    P.text((lx+lw-8, H-19), f'HRA {sex} reference', 11, A['mute'], 'rm')

    # Viewer.
    vx, vy = lx + lw + 6, 94
    vw, vh = PANEL_SIZE
    P.paste(viewport(F), (vx, vy))
    P.box((vx-1, vy-1, vw+2, vh+2), outline=A['red'], width=1)
    img, mapping = overview(p.slide, (202, 142))
    ox2, oy2 = vx + vw - img.width - 14, vy + 10
    P.box((ox2-4, oy2-4, img.width+8, img.height+8), fill='#ffffff', outline='#202020')
    P.paste(img, (ox2, oy2))
    draw_footprint(P, footprint_px(slides, p.slide, p, mapping, (ox2, oy2)), A['red'], 2, None, 14)
    um_px = p.width_um / vw
    um = bar_length(um_px, 110)
    L = um/um_px
    P.shade((vx+4, vy+vh-42, L+16, 38), '#ffffff', .55)
    P.d.line((vx+12, vy+vh-30, vx+12+L, vy+vh-30), fill='#7a7a7a', width=3)
    P.text((vx+12+L/2, vy+vh-14), length_label(um), 12, '#5a5a5a', 'mm')
    centre = np.asarray(viewport_center_colour(F))
    P.shade((vx+vw-150, vy+vh-46, 142, 38), '#6e6e6e', .82)
    P.text((vx+vw-79, vy+vh-35), f'{p.x_um:,.2f}, {p.y_um:,.2f} µm', 11, '#ffffff', 'mm')
    P.text((vx+vw-79, vy+vh-19), f'{centre[0]}, {centre[1]}, {centre[2]}', 11, '#ffffff', 'mm')

    # Below the viewer: the project's slides, with similarity as a small trace.
    by = vy + vh + 8
    bh = H - 8 - by
    P.box((vx, by, vw, bh), fill='#ffffff', outline=A['line'])
    groups = {}
    for i, sl in enumerate(slides):
        groups.setdefault(sl.tissue, []).append(i)
    head = f'Project  {len(slides)} images'
    if len(groups) < len(slides):
        head += f'  {len(groups)} tissues'
    P.text((vx+8, by+12), head, 12, A['ink'], 'lm', bold=True)
    sx1 = vx + vw - 60
    P.text((sx1+52, by+12), f"{F['score']:.3f}", 12, A['red'], 'rm', bold=True)
    spark(P, (sx1-200, by+5, 190, 14), F['scores'], F['t'], '#c4c4c4', '#3a3a3a', A['red'], lo=0.)
    P.text((sx1-210, by+12), 'Similarity', 11, A['mute'], 'rm')
    # One cell per tissue. With several slides per tissue the cell shows the slide in
    # use, or the tissue's first slide, with a count.
    cells = list(groups.items())
    n = len(cells)
    rows_n = 2 if n <= 24 else 4
    cols = -(-n//rows_n)
    cw, ch = (vw-12)/cols, (bh-28)/rows_n
    side = rows_n > 2            # compact cells: thumbnail left, name right
    for k, (tissue, members) in enumerate(cells):
        cx = vx + 6 + (k % cols)*cw
        cy = by + 24 + (k//cols)*ch
        on = p.slide in members
        i = p.slide if on else members[0]
        if side:
            tw = int(min(cw*.48, (ch-6)*1.5))
            img, _ = overview(i, (tw, int(ch-6)))
            P.paste(img, (cx + 3 + (tw-img.width)/2, cy + (ch-img.height)/2))
            words, lines, line = tissue.split(), [], ''
            room = cw - tw - 10
            for wd in words:
                trial = (line + ' ' + wd).strip()
                if P.width(trial, 9) > room and line:
                    lines.append(line)
                    line = wd
                else:
                    line = trial
            lines.append(line)
            lines = lines[:2]
            ty = cy + ch/2 - (len(lines)-1)*5.5
            for j, ln in enumerate(lines):
                P.text((cx + tw + 7, ty + j*11), ln, 9, A['blue'] if on else A['mute'], 'lm', bold=on)
        else:
            img, _ = overview(i, (int(cw-10), int(ch-18)))
            P.paste(img, (cx + (cw-img.width)/2, cy + 2 + (ch-18-img.height)/2))
            size = 10
            while size > 7 and P.width(tissue, size) > cw - 6:
                size -= 1
            P.text((cx+cw/2, cy+ch-9), tissue, size, A['blue'] if on else A['mute'], 'mm', bold=on)
        if on:
            P.box((cx+1, cy, cw-2, ch-1), outline=A['blue'], width=2)
        if len(members) > 1:
            P.text((cx+cw-5, cy+2), f'{len(members)}', 9, A['mute'], 'ra')
    return canvas


def viewport_center_colour(F):
    from .geometry import colour_crop
    from dataclasses import replace
    p = F['pose']
    small = colour_crop(F['slide'], replace(p, width_um=p.width_um/50), 4, 3)
    return small[1, 2]


# =====================================================================  B
B = dict(header='#4a4b57', tray='#f4f4f4', slot='#e3e3e3', line='#c9c9cd', ink='#26262b', mute='#77777f',
         green='#3d9a3d', select='#e9f3e9', orange='#f08a24')
B_ANAT = theme_key(map_bg=(255, 255, 255), map_organ=(206, 190, 198), map_line=(130, 130, 140),
                   accent_rgb=(61, 154, 61), block_rgb=(30, 30, 30))
ZOOM_COLOURS = {1.25: '#000000', 2.5: '#d62d2d', 5: '#f2a000', 10: '#e9e000', 20: '#2ea043', 40: '#2c5ad6'}
GROUPS = (('Nervous', ('Cerebral cortex', 'Cerebellum', 'Pituitary', 'Tibial nerve')),
          ('Cardiovascular', ('Heart', 'Auricular appendage', 'Coronary artery', 'Aorta', 'Tibial artery')),
          ('Respiratory', ('Lung',)),
          ('Digestive', ('Minor salivary gland', 'Esophageal mucous membrane', 'Esophageal muscle',
                         'Gastroesophageal junction', 'Stomach', 'Ileum', 'Transverse colon', 'Sigmoid colon',
                         'Liver', 'Pancreas')),
          ('Endocrine', ('Thyroid', 'Adrenal gland')), ('Lymphoid', ('Spleen',)),
          ('Urinary', ('Renal cortex', 'Renal medulla', 'Urinary bladder')),
          ('Reproductive', ('Prostate', 'Testis', 'Uterus', 'Ovary', 'Fallopian tube', 'Endocervix', 'Exocervix',
                            'Vagina', 'Breast')),
          ('Integument', ('Skin of lower leg', 'Skin of suprapubic region', 'Adipose tissue', 'Omentum',
                          'Skeletal muscle')))


def tray_order(slides):
    order = []
    for g, (name, tissues) in enumerate(GROUPS):
        for i, s in enumerate(slides):
            if s.tissue in tissues:
                order.append((g, i))
    rest = [(len(GROUPS), i) for i in range(len(slides)) if i not in {j for _, j in order}]
    return order + rest


def clinical(F):
    canvas = Image.new('RGB', SIZE, '#ffffff')
    P = Painter(canvas)
    p, s, slides = F['pose'], F['slide'], F['slides']
    W, H = SIZE
    P.box((0, 0, W, 34), fill=B['header'])
    P.text((14, 17), 'Bad Apple!!', 14, '#ffffff', 'lm', bold=True)
    P.text((104, 17), 'Slide Viewer', 14, '#c9cad3', 'lm')
    P.text((W-16, 17), f"Fr {F['frame_no']:04} / {F['total']}    {mmss(F['time_s'])}", 13, '#e6e6ec', 'rm')
    # Slide tray.
    tw = 356
    P.box((0, 34, tw, H-34), fill=B['tray'])
    P.box((0, 34, tw, 26), fill='#62636f')
    P.text((10, 47), f'GTEx  H&E  {len(slides)} slide(s)', 12, '#ffffff', 'lm')
    icon(P, 'tray', tw-16, 47, '#ffffff', 6)
    P.text((10, 72), 'View all memos', 11, B['mute'], 'lm')
    P.text((tw-10, 72), 'Sorted by: Organ system  Stain', 11, B['mute'], 'rm')
    order = tray_order(slides)
    cols = 2
    n = len(order)
    rows_n = -(-n//cols)
    top, bottom = 84, H - 40
    sh = min(58, (bottom-top)/rows_n - 6)
    sw = (tw - 56)/2 - 10
    last_group = None
    for k, (g, i) in enumerate(order):
        r, c = k % rows_n, k // rows_n
        x = 34 + c*(sw+26)
        y = top + r*(sh+6)
        on = i == p.slide
        if on:
            P.box((x-4, y-3, sw+8, sh+6), fill=B['select'], outline=B['green'], width=2)
        P.d.ellipse((x-24, y+sh/2-6, x-12, y+sh/2+6), outline='#a5a5ad', fill='#ffffff')
        glass_slide(P, (x, y, sw, sh), i, slides)
        P.text((x+sw+12, y+sh/2), str(k+1), 11, B['ink'], 'mm')
        if c == 0 and g != last_group:
            P.text((4, y-1), f'ID {g+1}', 9, B['mute'], 'la')
            last_group = g
    for k, (label, x) in enumerate((('Previous slide', 30), ('Next slide', tw/2+10))):
        P.box((x, H-32, tw/2-40, 20), fill='#8e95a3', radius=9)
        P.text((x+(tw/2-40)/2, H-22), label, 11, '#ffffff', 'mm')
    P.d.line((tw, 34, tw, H), fill=B['line'])
    # Tool strip.
    sx = tw + 1
    P.box((sx, 34, 30, H-34), fill='#ffffff')
    for k, name in enumerate(('close', 'arrow', 'pin', 'chat', 'layers', 'pen', 'expand', 'contrast', 'gear')):
        icon(P, name, sx+15, 56 + k*30 + (40 if k >= 5 else 0), '#5f6068', 7)
    P.box((sx+30, 34, 4, H-34), fill='#b9b9bf')
    # Viewer.
    vx, vy = sx + 40, 40
    vw, vh = PANEL_SIZE
    view = viewport(F)
    P.paste(view, (vx, vy))
    # Label and thumbnail with heat map, top right.
    lab = (vx+vw-340, vy+8, 96, 96)
    slide_label(P, lab, s)
    tw2, th2 = 228, 150
    img, mapping = overview(p.slide, (tw2, th2))
    tx, ty = vx+vw-8-tw2, vy+8
    P.box((tx-2, ty-2, tw2+4, th2+4), fill='#ffffff', outline='#8d8d92')
    P.paste(img, (tx + (tw2-img.width)//2, ty + (th2-img.height)//2))
    origin = (tx + (tw2-img.width)//2, ty + (th2-img.height)//2)
    for hp, hmag in F['history']:
        if hp.slide == p.slide:
            pts = footprint_px(slides, hp.slide, hp, mapping, origin)
            P.d.polygon(pts, outline=ZOOM_COLOURS[nearest_objective(hmag)])
    draw_footprint(P, footprint_px(slides, p.slide, p, mapping, origin), '#000000', 2, '#ffffff', 12)
    # Zoom slider and colour-coded zoom control, bottom left.
    zx, zy = vx + 10, vy + vh - 170
    P.box((zx, zy, 26, 160), fill='#ffffff', outline='#8d8d92', radius=3)
    near = nearest_objective(F['mag'])
    frac = (np.log(F['mag']) - np.log(1.25)) / (np.log(40) - np.log(1.25))
    knob = zy + 150 - 140*float(np.clip(frac, 0, 1))
    P.d.line((zx+13, zy+10, zx+13, zy+150), fill='#c7c7cc', width=3)
    P.box((zx+3, knob-7, 20, 14), fill='#ffffff', outline='#6a6a70', radius=2)
    P.text((zx+13, zy+168), f"{F['mag']:.1f}x", 11, B['ink'], 'mm')
    cx0 = zx + 34
    for k, o in enumerate(OBJECTIVES):
        yy = zy + 8 + k*24
        on = o == near
        P.box((cx0, yy, 40, 20), fill='#2d2d33' if on else '#ffffff', outline='#8d8d92')
        P.box((cx0, yy, 5, 20), fill=ZOOM_COLOURS[o])
        P.text((cx0+23, yy+10), f'{o:g}x', 11, '#ffffff' if on else B['ink'], 'mm')
    # Rotation dial showing the field's angle.
    rx, ry, rr = cx0 + 92, vy + vh - 64, 40
    P.d.ellipse((rx-rr, ry-rr, rx+rr, ry+rr), fill='#ffffff', outline='#8d8d92', width=2)
    P.d.ellipse((rx-rr+12, ry-rr+12, rx+rr-12, ry+rr-12), outline='#b5b5bb')
    a = np.deg2rad(p.angle_deg - 90)
    P.d.line((rx, ry, rx+np.cos(a)*(rr-6), ry+np.sin(a)*(rr-6)), fill=B['green'], width=3)
    P.d.ellipse((rx-4, ry-4, rx+4, ry+4), fill='#5f6068')
    P.text((rx, ry+rr+10), f'{p.angle_deg:+.0f}°', 11, B['ink'], 'mm')
    # Magnifier, bottom right: twice the magnification of the region beneath it.
    mr = 74
    mx, my = vx + vw - mr - 16, vy + vh - mr - 50
    region = np.asarray(Image.fromarray(view).crop((mx-mr/2-vx, my-mr/2-vy, mx+mr/2-vx, my+mr/2-vy))
                        .resize((2*mr, 2*mr), Image.Resampling.LANCZOS))
    mask = Image.new('L', (2*mr, 2*mr), 0)
    from PIL import ImageDraw
    ImageDraw.Draw(mask).ellipse((0, 0, 2*mr-1, 2*mr-1), fill=255)
    canvas.paste(Image.fromarray(region), (int(mx-mr), int(my-mr)), mask)
    P.d.ellipse((mx-mr, my-mr, mx+mr, my+mr), outline='#2d2d33', width=3)
    P.d.line((mx+mr*.7, my+mr*.7, mx+mr*.7+14, my+mr*.7+14), fill='#2d2d33', width=5)
    # Scale bar.
    um_px = p.width_um / vw
    um = bar_length(um_px, 90)
    L = um/um_px
    bx, byy = vx + vw - 16 - L, vy + vh - 18
    P.box((bx-6, byy-24, L+12, 30), fill='#ffffff', outline='#8d8d92')
    P.text((bx+L/2, byy-16), f'{near:g}x', 10, B['ink'], 'mm')
    P.d.line((bx, byy-4, bx+L, byy-4), fill='#26262b', width=2)
    P.d.line((bx, byy-8, bx, byy), fill='#26262b')
    P.d.line((bx+L, byy-8, bx+L, byy), fill='#26262b')
    P.text((bx+L/2, byy+3), length_label(um), 10, B['ink'], 'mm')
    # Case panel below the viewer: reference, body, organ, details.
    cy = vy + vh + 6
    ch = H - cy - 6
    P.box((vx, cy, vw, ch), fill='#fafafa', outline=B['line'])
    rw = round((ch-24)*4/3)
    P.text((vx+8, cy+11), 'Reference', 11, B['mute'], 'lm')
    ref = Image.fromarray(F['original']).convert('RGB').resize((rw, ch-24), Image.Resampling.LANCZOS)
    P.paste(ref, (vx+8, cy+20))
    bx0 = vx + 16 + rw
    bw0 = 150
    P.text((bx0, cy+11), 'Specimen site', 11, B['mute'], 'lm')
    img, site, sex = anatomy_card(s.tissue, (bw0, ch-24), B_ANAT, 'body')
    P.paste(img, (bx0, cy+20))
    site_marker(P, site, (bx0, cy+20), B['green'], '#ffffff')
    ox0 = bx0 + bw0 + 6
    ow0 = 300
    img, site, sex = anatomy_card(s.tissue, (ow0, ch-24), B_ANAT, 'organ', 25.)
    P.paste(img, (ox0, cy+20))
    site_marker(P, site, (ox0, cy+20), B['green'], '#ffffff')
    ix = ox0 + ow0 + 14
    rows = [('Specimen', s.spec['specimen']), ('Tissue', s.tissue), ('Site', site_caption(F)),
            ('Atlas', f'HRA {sex} body'), ('Channel', CHANNELS[p.channel].capitalize()),
            ('Field', f'{p.width_um/1000:.2f} × {p.width_um*.75/1000:.2f} mm'),
            ('Similarity', f"{F['score']:.3f}")]
    for k, (a_, b_) in enumerate(rows):
        P.text((ix, cy+14+k*19), a_, 11, B['mute'], 'lm')
        P.text((ix+72, cy+14+k*19), b_, 11, B['ink'], 'lm', bold=a_ == 'Tissue')
    spark(P, (ix, cy+ch-36, vx+vw-ix-12, 24), F['scores'], F['t'], '#c9c9cd', '#55555c', B['green'], lo=0.)
    return canvas


# =====================================================================  C
C = dict(window='#0b0b0d', title='#232331', tab='#7d86c9', ink='#f0f0f4', mute='#a0a0ac', red='#ff3b30',
         line='#3a3a46')
C_ANAT = theme_key(map_bg=(14, 14, 17), map_organ=(120, 104, 118), map_line=(200, 205, 215),
                   accent_rgb=(255, 92, 140), block_rgb=(255, 255, 255))


def window(P, box, title, alpha=.82):
    x, y, w, h = box
    P.shade((x, y, w, h), C['window'], alpha)
    P.box((x, y, w, 20), fill=C['title'])
    P.text((x+8, y+10), title, 11, C['ink'], 'lm')
    icon(P, 'close', x+w-11, y+10, C['mute'], 6)
    P.box((x, y, w, h), outline=C['line'])
    return x, y+20, w, h-20


def scanner(F):
    p, s, slides = F['pose'], F['slide'], F['slides']
    W, H = SIZE
    # The field fills the screen: the 1440 x 1080 canvas is exactly 4:3.
    canvas = Image.fromarray(viewport(F, SIZE))
    P = Painter(canvas)
    P.shade((0, 0, W, 26), '#000000', .9)
    P.box((4, 3, 22, 20), fill='#1d1d29', outline='#6a6fa8')
    P.text((15, 13), 'B', 13, '#c9ccf2', 'mm', bold=True)
    label = f"{s.spec['specimen']}  {s.tissue}"
    tw = P.width(label, 12) + 40
    P.box((30, 3, tw, 20), fill=C['tab'])
    P.text((38, 13), label, 12, '#ffffff', 'lm')
    icon(P, 'close', 30+tw-12, 13, '#ffffff', 6)
    P.box((32+tw, 3, 20, 20), fill='#1d1d29', outline='#555')
    P.text((42+tw, 13), '+', 14, '#ffffff', 'mm')
    P.text((W-14, 13), f"Sim {F['score']:.3f}     {F['frame_no']:04} / {F['total']}     {mmss(F['time_s'])}",
           12, C['ink'], 'rm')
    # Browse window: thumbnails of the cohort.
    bx, by, bw, bh = window(P, (12, 36, 360, 250), 'Browse  GTEx H&E')
    n = len(slides)
    cols = 5
    rows_n = -(-n//cols)
    cw, ch = (bw-8)/cols, (bh-8)/rows_n
    for i in range(n):
        cx, cy = bx+4+(i % cols)*cw, by+4+(i//cols)*ch
        img, _ = overview(i, (int(cw-6), int(ch-14)), .95)
        P.paste(img, (cx+(cw-img.width)/2, cy+1))
        on = i == p.slide
        P.text((cx+cw/2, cy+ch-6), slides[i].spec['specimen'][5:], 8, '#ffffff' if on else C['mute'], 'mm')
        if on:
            P.box((cx+1, cy, cw-2, ch-1), outline=C['tab'], width=2)
    # Slide image window: the whole glass slide with its label.
    gx, gy, gw, gh = window(P, (12, 296, 360, 150), 'Slide image')
    glass_slide(P, (gx+8, gy+10, gw-16, gh-20), p.slide, slides, light=True)
    # Atlas window: HRA body and organ.
    ax, ay, aw, ah = window(P, (12, 456, 360, 340), f'Atlas  {s.tissue}')
    img, site, sex = anatomy_card(s.tissue, (130, ah-26), C_ANAT, 'body')
    P.paste(img, (ax+4, ay+4))
    site_marker(P, site, (ax+4, ay+4), '#ffffff', '#000000')
    img, site2, _ = anatomy_card(s.tissue, (aw-142, ah-26), C_ANAT, 'organ', 25.)
    P.paste(img, (ax+138, ay+4))
    site_marker(P, site2, (ax+138, ay+4), '#ffffff', '#000000')
    P.text((ax+6, ay+ah-11), f'HRA {sex} reference', 10, C['mute'], 'lm')
    P.text((ax+aw-6, ay+ah-11), site_caption(F), 10, C['mute'], 'rm')
    # Reference window, top right.
    rx, ry, rw, rh = window(P, (W-12-300, 36, 300, 245), 'Reference')
    ref = Image.fromarray(F['original']).convert('RGB').resize((rw-8, rh-8), Image.Resampling.LANCZOS)
    P.paste(ref, (rx+4, ry+4))
    # Tool strip and lens (magnification) window on the right edge.
    tx = W - 12 - 30
    P.shade((tx, 292, 30, 230), C['window'], .85)
    P.box((tx, 292, 30, 230), outline=C['line'])
    for k, (name, col) in enumerate((('arrow', '#ffffff'), ('pen', '#f6c23e'), ('ellipse', '#f6c23e'),
                                     ('polygon', '#f6c23e'), ('ruler', '#ffffff'), ('pin', C['red']),
                                     ('folder', '#f6c23e'), ('gear', '#ffffff'))):
        icon(P, name, tx+15, 308+k*28, col, 7)
    lx, ly, lw, lh = window(P, (W-12-30-8-120, 292, 120, 196), 'Lens')
    near = nearest_objective(F['mag'])
    for k, o in enumerate(OBJECTIVES):
        yy = ly + 6 + k*28
        on = o == near
        if on:
            P.box((lx+6, yy, lw-12, 24), fill='#3e4691', outline='#9aa2ee')
        P.text((lx+lw/2, yy+12), f'x{o:g}', 13, '#ffffff' if on else C['mute'], 'mm', bold=on)
    # View widget: rotation compass.
    vx_, vy_, vr = W - 12 - 30 - 8 - 60, 548, 44
    P.d.ellipse((vx_-vr, vy_-vr, vx_+vr, vy_+vr), fill='#15151c', outline='#6a6fa8', width=2)
    a = np.deg2rad(p.angle_deg - 90)
    P.d.line((vx_, vy_, vx_+np.cos(a)*(vr-8), vy_+np.sin(a)*(vr-8)), fill='#9aa2ee', width=3)
    P.text((vx_, vy_+vr-14), f'{p.angle_deg:+.0f}°', 11, '#ffffff', 'mm')
    # Map window, bottom right.
    mx, my, mw, mh = window(P, (W-12-330, H-12-220, 330, 220), 'Map')
    img, mapping = overview(p.slide, (mw-12, mh-12))
    origin = (mx + (mw-img.width)//2, my + (mh-img.height)//2)
    P.paste(img, origin)
    draw_footprint(P, footprint_px(slides, p.slide, p, mapping, origin), C['red'], 2, '#000000', 12)
    # Scale bar, bottom left of the viewing area.
    um_px = p.width_um / W
    um = bar_length(um_px, 140)
    L = um/um_px
    sx, sy = 392, H - 34
    P.shade((sx-8, sy-20, L+16, 32), '#000000', .7)
    P.d.line((sx, sy, sx+L, sy), fill='#ffffff', width=3)
    P.text((sx+L/2, sy-10), length_label(um), 11, '#ffffff', 'mm')
    info = (f"{F['mag']:.1f}x   {CHANNELS[p.channel].capitalize()}   "
            f"Field {p.width_um/1000:.2f} × {p.width_um*.75/1000:.2f} mm")
    P.shade((sx+L+14, sy-20, P.width(info, 12)+20, 32), '#000000', .7)
    P.text((sx+L+24, sy-4), f"{F['mag']:.1f}x   {CHANNELS[p.channel].capitalize()}   "
           f"Field {p.width_um/1000:.2f} × {p.width_um*.75/1000:.2f} mm", 12, '#ffffff', 'lm')
    return canvas


LAYOUTS = dict(workstation=workstation, clinical=clinical, scanner=scanner)
