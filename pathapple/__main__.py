import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .inputs import ROOT, SOURCES, fetch, verify


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def ffmpeg_version():
    try:
        return subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]
    except FileNotFoundError:
        return 'not installed; frames decoded by OpenCV ' + cv2.__version__


def slide_meta(s):
    """Public identifiers, coded site and image geometry only: no demographics and no
    free-text pathology notes are exported."""
    return dict(key=s.key, specimen=s.spec['specimen'], tissue=s.tissue, snomed=s.spec['snomed'],
                fine_um_per_px=s.fine_um, plane_um_per_px=s.base_um, plane_shape=list(s.planes.shape[1:]),
                glass_rgb=[float(v) for v in s.glass], stain_vectors=np.asarray(s.stains).round(5).tolist())


def match(args):
    from .geometry import CHANNELS, Pose, footprint, load_slides
    from .media import frames, frame_count
    from .report import report
    from .search import pipeline
    verify()
    originals, indices, rate = frames(args.start, args.seconds, args.fps)
    begin = time.perf_counter()
    slides = load_slides()
    load_time = time.perf_counter() - begin
    print(f'Loaded {len(slides)} slides in {load_time:.1f} s', flush=True)
    args.output.mkdir(exist_ok=True, parents=True)
    key = dict(seed=args.seed, samples=args.samples, coarse_width=args.coarse_width, source_frames=indices,
               slides=hashlib.sha256(json.dumps(SOURCES['slides']).encode()).hexdigest())
    baseline, selected, stats = pipeline(slides, originals, args.seed, args.samples, args.coarse_width,
                                         args.device, args.output/'retrieval.json.gz', key, args.workers)
    records = []
    for t, (a, b) in enumerate(zip(baseline, selected)):
        for m in (a, b):
            p = Pose(**m['pose'])
            m.update(slide_key=slides[p.slide].key, tissue=slides[p.slide].tissue, channel=CHANNELS[p.channel],
                     corners_fine_px=footprint(slides[p.slide], p, slides[p.slide].fine_um))
        records.append(dict(frame=t, time_s=args.start + t/args.fps, source_frame=indices[t],
                            coarse_ncc=a, silhouette=b))
    stats['load_seconds'] = load_time
    for name, ms in (('coarse_ncc', baseline), ('silhouette', selected)):
        stats[name] = {k: float(np.mean([m['metrics'][k] for m in ms])) for k in ms[0]['metrics']}
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ('cmd', 'output')}
    record = dict(schema=1, config=config,
                  dataset=dict(video=SOURCES['video'], slides=SOURCES['slides']),
                  slides=[slide_meta(s) for s in slides], source_fps=rate, source_frame_count=frame_count(),
                  environment=dict(machine=platform.machine(), os=platform.platform(), python=platform.python_version(),
                                   ffmpeg=ffmpeg_version(),
                                   packages={k: importlib.metadata.version(k) for k in (
                                       'numpy', 'opencv-python-headless', 'pillow', 'torch', 'pydicom', 'trimesh',
                                       'moderngl')}),
                  benchmark=stats, frames=records)
    write_json(args.output/'matches.json', record)
    report(args.output, record)
    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, indent=2), flush=True)
    return record


def render(output, layout, style, to=None):
    from .geometry import load_slides
    from .media import frames
    from .video import encode
    verify()
    record = json.loads((output/'matches.json').read_text())
    if record['schema'] != 1 or record['dataset']['video'] != SOURCES['video']:
        raise ValueError('Saved match data references a different source or an unsupported schema')
    pinned = {s['key']: s for s in SOURCES['slides']}
    if [s['key'] for s in record['dataset']['slides']] != [s['key'] for s in SOURCES['slides']] or \
            any(pinned[s['key']] != s for s in record['dataset']['slides']):
        raise ValueError('The run was matched against a different slide bank than sources.json pins')
    c = record['config']
    originals, indices, rate = frames(c['start'], c['seconds'], c['fps'])
    if rate != record['source_fps'] or indices != [r['source_frame'] for r in record['frames']]:
        raise ValueError('Source frame mapping differs from the saved run')
    out = to or output
    out.mkdir(parents=True, exist_ok=True)
    return encode(out, record, load_slides(), originals, layout, style)


def main():
    from .layouts import LAYOUTS
    from .styles import STYLES
    cli = argparse.ArgumentParser(description='Match Bad Apple!! to GTEx histology slides.')
    commands = cli.add_subparsers(dest='cmd', required=True)
    commands.add_parser('fetch', help='Download and hash-check the pinned slides, anatomy and media')
    for name in ('run', 'match', 'render'):
        parser = commands.add_parser(name)
        parser.add_argument('--output', type=Path, default=ROOT/'output')
        if name != 'match':
            parser.add_argument('--layout', choices=sorted(LAYOUTS), default='workstation')
            parser.add_argument('--style', choices=sorted(STYLES), default='adaptive')
        if name == 'render':
            parser.add_argument('--to', type=Path, help='Write the video to this directory instead')
        else:
            parser.add_argument('--start', type=float, default=8.)
            parser.add_argument('--seconds', type=float, default=7.)
            parser.add_argument('--fps', type=int, default=10)
            parser.add_argument('--seed', type=int, default=1513)
            parser.add_argument('--samples', type=int, default=16384,
                                help='Proposed fields per slide and magnification band')
            parser.add_argument('--coarse-width', type=int, default=32,
                                help='Retrieval thumbnail width (multiple of 4)')
            parser.add_argument('--workers', type=int, default=None, help='Refinement workers')
            parser.add_argument('--device', default='auto', help='auto, mps, cuda or cpu')
    args = cli.parse_args()
    cv2.setNumThreads(1)
    if args.cmd == 'fetch':
        fetch()
    elif args.cmd == 'render':
        render(args.output, args.layout, args.style, args.to)
    else:
        if args.fps <= 0 or args.samples < 4 or args.seed < 0 or args.coarse_width % 4 or args.coarse_width < 8:
            cli.error('fps must be positive, samples >= 4, seed >= 0, coarse width a multiple of 4 and >= 8')
        match(args)
        if args.cmd == 'run':
            # A fresh process renders: this one has used the Apple GPU, which cannot be forked.
            subprocess.run([sys.executable, '-m', 'pathapple', 'render', '--output', str(args.output),
                            '--layout', args.layout, '--style', args.style], check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
