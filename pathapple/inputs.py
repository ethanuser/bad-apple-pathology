"""Pinned inputs, fetched individually from public buckets and checked before use."""
import hashlib
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets'
SOURCES = json.loads((ROOT / 'sources.json').read_text())
HRA_MESHES = json.loads((ROOT / 'data' / 'hra' / 'meshes.json').read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inputs():
    yield ASSETS / 'badapple.mp4', SOURCES['video']
    for slide in SOURCES['slides']:
        for level in ('fine', 'overview'):
            yield ASSETS / slide['key'] / f'{level}.dcm', slide['levels'][level]
    for item in HRA_MESHES + SOURCES.get('fonts', []):
        yield ASSETS / item['path'], item


def fetch_one(item):
    path, spec = item
    if path.exists() and ('sha256' not in spec or digest(path) == spec['sha256']):
        return path
    path.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        partial = Path(temp.name)
    try:
        subprocess.run(['curl', '-fsSL', '--retry', '3', '--max-time', '300',
                        '-o', str(partial), spec['url']], check=True)
        if 'sha256' in spec and digest(partial) != spec['sha256']:
            raise ValueError(f'Checksum mismatch for {path.name} in {path.parent.name}')
        partial.replace(path)
        print(f'Verified {path.relative_to(ASSETS)}', flush=True)
    finally:
        partial.unlink(missing_ok=True)
    return path


def fetch(pin=False):
    items = list(inputs())
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch_one, items))
    if pin:
        # Record hashes of newly selected levels the first time they are fetched.
        for path, spec in items:
            spec.setdefault('sha256', digest(path))
        (ROOT/'sources.json').write_text(json.dumps(SOURCES, indent=2) + '\n')


def verify():
    for path, spec in inputs():
        if not path.is_file() or digest(path) != spec['sha256']:
            raise ValueError(f'Missing or altered input {path}. Run python -m pathapple fetch.')
