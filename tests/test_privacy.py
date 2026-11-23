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
