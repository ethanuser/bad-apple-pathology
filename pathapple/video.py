"""Encode a saved run as video: one workstation frame per match, piped to FFmpeg with
the song's audio. Frames are drawn by forked workers in contiguous blocks, so each
worker mostly stays on one decoded slide."""
import multiprocessing
import os
import subprocess
import tempfile
from collections import deque
from pathlib import Path

from PIL import Image

from .geometry import Pose
from .layouts import LAYOUTS, SIZES
from .media import VIDEO
from .render import frame_info, magnification

BLOCK = 12
_JOB = {}


def frames_for(record, slides, originals, style):
    rows = record['frames']
    scores = [r['silhouette']['metrics']['score'] for r in rows]
    history = [(Pose(**r['silhouette']['pose']), magnification(Pose(**r['silhouette']['pose']))) for r in rows]
    total = record['source_frame_count']
    for t, r in enumerate(rows):
        yield t, dict(match=r['silhouette'], original=originals[t], frame_no=r['source_frame']+1, total=total,
                      time_s=r['time_s'], scores=scores, t=t, style=style, history=history[max(0, t-300):t])


def draw(t):
    j = _JOB
    args = j['frames'][t]
    F = frame_info(j['slides'], args['match'], args['original'], args['frame_no'], args['total'], args['time_s'],
                   args['scores'], args['t'], args['style'], args['history'])
    return LAYOUTS[j['layout']](F).tobytes()


def _render_block(bounds):
    return [draw(t) for t in range(*bounds)]


def _prepare_cards(layout, sample_frames):
    """Run in a spawned process: draw one frame per tissue so every anatomy card the
    layout needs is rendered with OpenGL and cached on disk."""
    from .geometry import load_slides
    slides = load_slides()
    for args in sample_frames:
        F = frame_info(slides, args['match'], args['original'], args['frame_no'], args['total'], args['time_s'],
                       args['scores'], args['t'], args['style'], args['history'])
        LAYOUTS[layout](F)


def workers():
    n = os.cpu_count() or 1
    for name in ('SLURM_CPUS_PER_TASK', 'CPU_LIMIT'):
        if os.environ.get(name):
            n = max(1, int(float(os.environ[name])))
    return max(1, min(6, n))     # each worker holds up to two decoded slides


def encode(output, record, slides, originals, layout='workstation', style='adaptive', name=None):
    conf = record['config']
    SIZE = SIZES[layout]
    name = name or ('bad-apple-pathology.mp4' if layout == 'workstation' else f'bad-apple-pathology-{layout}.mp4')
    frames = dict(frames_for(record, slides, originals, style))
    seen, samples = set(), []
    for t, args in frames.items():
        tissue = slides[args['match']['pose']['slide']].tissue
        if tissue not in seen:
            seen.add(tissue)
            samples.append(args)
    with multiprocessing.get_context('spawn').Pool(1) as pool:
        pool.apply(_prepare_cards, (layout, samples))
    path = output/name
    with tempfile.NamedTemporaryFile(dir=output, suffix='.mp4', delete=False) as f:
        temp = Path(f.name)
    args = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{SIZE[0]}x{SIZE[1]}', '-r', str(conf['fps']), '-i', 'pipe:0',
            '-ss', str(conf['start']), '-i', str(VIDEO), '-map', '0:v:0', '-map', '1:a:0?',
            '-t', str(len(frames)/conf['fps']), '-c:v', 'libx264', '-preset', 'fast', '-crf', '16',
            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart',
            '-metadata', 'title=Bad Apple!! / Closest Histology Slide',
            '-metadata', 'comment=Histology: GTEx Consortium via NCI Imaging Data Commons, CC BY 4.0. '
            'Anatomy: Human Reference Atlas 3D reference organs, CC BY 4.0. Shadow PV: Anira. '
            'Music: Alstroemeria Records, nomico, ZUN. See accompanying credits.', str(temp)]
    blocks = [(lo, min(lo+BLOCK, len(frames))) for lo in range(0, len(frames), BLOCK)]
    n = workers()
    _JOB.update(frames=frames, slides=slides, layout=layout)
    done = 0
    try:
        with subprocess.Popen(args, stdin=subprocess.PIPE) as proc:
            try:
                def write(chunk):
                    nonlocal done
                    for raw in chunk:
                        if done == 0:
                            Image.frombytes('RGB', SIZE, raw).save(output/f'preview-{layout}.jpg', quality=94)
                        proc.stdin.write(raw)
                        done += 1
                    if done % 240 < BLOCK or done == len(frames):
                        print(f'Rendered {done}/{len(frames)} frames', flush=True)
                if n > 1:
                    with multiprocessing.get_context('fork').Pool(n) as pool:
                        pending = deque()
                        for blk in blocks:
                            pending.append(pool.apply_async(_render_block, (blk,)))
                            if len(pending) >= 2*n:
                                write(pending.popleft().get())
                        while pending:
                            write(pending.popleft().get())
                else:
                    for blk in blocks:
                        write(_render_block(blk))
            finally:
                proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError('FFmpeg failed to encode video')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
        _JOB.clear()
    print(f'Rendered {path}', flush=True)
    return path
