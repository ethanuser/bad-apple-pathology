"""Only public specimen IDs, coded anatomic sites and image geometry are pinned or
exported: no age, sex or other donor fields, and no free-text pathology notes."""
import json
import unittest

from pathapple.inputs import ROOT, SOURCES

FORBIDDEN = {'age', 'sex', 'patientage', 'patientsex', 'death', 'hardy', 'notes', 'description', 'diagnosis'}


def keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k.lower()
            yield from keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from keys(v)


class Privacy(unittest.TestCase):
    def test_sources(self):
        self.assertEqual(set(keys(SOURCES['slides'])) & FORBIDDEN, set())

    def test_saved_run(self):
        path = ROOT / 'output' / 'matches.json'
        if not path.exists():
            self.skipTest('no run saved')
        record = json.loads(path.read_text())
        self.assertEqual(set(keys(record['slides'])) & FORBIDDEN, set())
        self.assertEqual(set(keys(record['dataset'])) & FORBIDDEN, set())


if __name__ == '__main__':
    unittest.main()


class Content(unittest.TestCase):
    """No genital imagery: those tissues are not in the bank, and the 3D body's skin
    shell has the groin removed (the MRI/CT edition crops the pelvis the same way)."""
    def test_no_excluded_tissues(self):
        self.assertEqual({s['tissue'] for s in SOURCES['slides']} & {'Testis', 'Vagina'}, set())

    def test_groin_removed_from_body(self):
        from pathapple.anatomy import GROIN_HALF_WIDTH, HRA, body
        from pathapple.inputs import HRA_MESHES
        import numpy as np
        if not all((HRA.parent / m['path']).exists() for m in HRA_MESHES):
            self.skipTest('anatomy not fetched')
        for sex in ('female', 'male'):
            b = body(sex)
            pelvis = b.points('pelvis')
            lo, mid = pelvis[:, 1].min(), np.median(pelvis[:, 0])
            for m in b.parts('skin'):
                c = m.v[m.f].mean(axis=1)
                inside = (np.abs(c[:, 0] - mid) < GROIN_HALF_WIDTH) & (c[:, 1] > lo - .12) & (c[:, 1] < lo + .08)
                self.assertFalse(inside.any(), f'{sex} skin keeps {inside.sum()} groin faces')
