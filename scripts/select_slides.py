"""Choose GTEx H&E slides from the IDC index and pin them in sources.json.

For each requested tissue, a seeded draw of candidate slides is ranked by the share
of the 8 µm overview that holds tissue, and the best are pinned. Only public study
IDs, the coded anatomic site and image geometry are recorded: no demographics and
no free-text pathology notes."""
import argparse
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathapple.slides import decode_level, tissue_mask  # noqa: E402

BUCKET = 'https://idc-open-data.s3.amazonaws.com'


def level_rows(si, series):
    rows = si[(si.SeriesInstanceUID == series) & si.ImageType.map(lambda t: 'VOLUME' in t)]
    return rows.sort_values('PixelSpacing_0')


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--per-tissue', type=int, default=1)
    cli.add_argument('--candidates', type=int, default=4)
    cli.add_argument('--tissues', nargs='*')
    cli.add_argument('--seed', type=int, default=1513)
    cli.add_argument('--out', type=Path, default=ROOT/'sources.json')
    args = cli.parse_args()
    from idc_index import IDCClient
    c = IDCClient()
    c.fetch_index('sm_index')
    c.fetch_index('sm_instance_index')
    series = c.index[c.index.collection_id == 'gtex'][['SeriesInstanceUID', 'crdc_series_uuid', 'PatientID', 'license_short_name', 'source_DOI']]
    sm = c.sm_index.merge(series, on='SeriesInstanceUID')
    si = c.sm_instance_index
    tissues = args.tissues or sorted(sm.primaryAnatomicStructure_CodeMeaning.unique())
    rng = np.random.default_rng(args.seed)
    cache = ROOT/'assets'/'_select'
    cache.mkdir(parents=True, exist_ok=True)
    pinned = []
    for tissue in tissues:
        pool = sm[sm.primaryAnatomicStructure_CodeMeaning == tissue].sort_values('ContainerIdentifier')
        picks = pool.iloc[rng.choice(len(pool), min(len(pool), args.candidates), replace=False)]
        ranked = []
        for _, row in picks.iterrows():
            levels = level_rows(si, row.SeriesInstanceUID)
            # Want the ~2 µm level for matching and the ~8 µm level as the overview.
            fine = levels[(levels.PixelSpacing_0 > .0015) & (levels.PixelSpacing_0 < .0025)]
            coarse = levels[(levels.PixelSpacing_0 > .006) & (levels.PixelSpacing_0 < .010)]
            if fine.empty or coarse.empty:
                continue
            fine, coarse = fine.iloc[0], coarse.iloc[0]
            path = cache/f'{row.ContainerIdentifier}-overview.dcm'
            url = f'{BUCKET}/{row.crdc_series_uuid}/{coarse.crdc_instance_uuid}.dcm'
            if not path.exists():
                subprocess.run(['curl', '-fsSL', '--retry', '3', '-o', str(path), url], check=True)
            rgb = decode_level(path.read_bytes())
            share = float(tissue_mask(rgb).mean())
            ranked.append((share, row, fine, coarse))
            print(f'{tissue:28s} {row.ContainerIdentifier:18s} tissue {share:.1%}', flush=True)
        ranked.sort(key=lambda r: -r[0])
        for share, row, fine, coarse in ranked[:args.per_tissue]:
            levels = {}
            for name, lv in (('fine', fine), ('overview', coarse)):
                url = f'{BUCKET}/{row.crdc_series_uuid}/{lv.crdc_instance_uuid}.dcm'
                levels[name] = dict(url=url, bytes=int(lv.instance_size),
                                    um_per_px=float(lv.PixelSpacing_0)*1000,
                                    width=int(lv.TotalPixelMatrixColumns), height=int(lv.TotalPixelMatrixRows))
            overview = cache/f'{row.ContainerIdentifier}-overview.dcm'
            levels['overview']['sha256'] = hashlib.sha256(overview.read_bytes()).hexdigest()
            pinned.append(dict(key=row.ContainerIdentifier.lower(), specimen=row.ContainerIdentifier,
                               subject=row.PatientID, tissue=tissue,
                               snomed=row.primaryAnatomicStructure_code_designator_value_str,
                               series_uid=row.SeriesInstanceUID, licence=row.license_short_name,
                               doi=row.source_DOI, objective=float(row.ObjectiveLensPower),
                               tissue_share=round(share, 4), levels=levels))
    old = json.loads(args.out.read_text()) if args.out.exists() else {}
    old['slides'] = pinned
    args.out.write_text(json.dumps(old, indent=2) + '\n')
    print(f'Pinned {len(pinned)} slides', flush=True)


if __name__ == '__main__':
    main()
