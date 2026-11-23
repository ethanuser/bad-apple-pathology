"""Summarise a run: both matchers' similarity components, sources used and timing."""
import shutil
from collections import Counter

from .inputs import ROOT


def report(output, record):
    cfg, bench, frames = record['config'], record['benchmark'], record['frames']
    rows = []
    for title, key in (('Grayscale NCC (first retrieval)', 'coarse_ncc'), ('Contour search + temporal refinement', 'silhouette')):
        m = bench[key]
        slides = [r[key]['slide_key'] for r in frames]
        changes = sum(a != b for a, b in zip(slides, slides[1:]))
        rows.append(f"| {title} | {m['miou']:.3f} | {m['boundary_f1']:.3f} | {m['detail']:.3f} | {m['ncc']:.3f} | "
                    f"{m['score']:.3f} | {len(set(slides))} | {changes} |")
    tissues = Counter(r['silhouette']['tissue'] for r in frames)
    channels = Counter(r['silhouette']['channel'] for r in frames)
    out = [
        '# Matching results', '',
        f"{len(frames)} frames from {cfg['start']:.1f} to {cfg['start']+cfg['seconds']:.1f} s at {cfg['fps']} fps, "
        f"searched against {len(record['slides'])} GTEx H&E slides ({len({s['tissue'] for s in record['slides']})} tissues). "
        f"Coarse retrieval scored {bench['crops']:,} stain-channel fields on {bench['device'].upper()} "
        f"in {bench['coarse_seconds']:.0f} s with {bench['coarse_width']}-pixel thumbnails.", '',
        '| Matcher | Balanced IoU | Boundary F1 | Thin detail | Grayscale NCC | Similarity | Slides used | Slide changes |',
        '|---|---:|---:|---:|---:|---:|---:|---:|', *rows, '',
        'Similarity is 28% balanced IoU, 27% boundary F1 (2 pixels at 96 × 72), 20% thin detail, 12.5% positive '
        'global NCC and 12.5% positive regional NCC, measured on the silhouette that sets the displayed field\'s tone.', '',
        'Tissues in the selected sequence: ' + ', '.join(f'{t} {n}' for t, n in tissues.most_common()) + '.', '',
        'Stain channels: ' + ', '.join(f'{c} {n}' for c, n in channels.most_common()) + '.', '',
        'Every pose, threshold and slide is in [matches.json](matches.json). Credits are in [CREDITS.md](CREDITS.md).', '']
    (output/'comparison.md').write_text('\n'.join(out))
    shutil.copyfile(ROOT/'CREDITS.md', output/'CREDITS.md')
