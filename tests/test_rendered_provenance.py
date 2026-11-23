"""Decode the rendered video and re-draw each sampled frame's viewport independently
from matches.json and the pinned slides: the encoded pixels must match the logged
pose within a lossy-codec tolerance."""
import json
import unittest

import cv2
import numpy as np

from pathapple.inputs import ROOT

OUT = ROOT / 'output'


@unittest.skipUnless((OUT / 'bad-apple-pathology.mp4').exists() and (OUT / 'matches.json').exists(), 'no render')
class Provenance(unittest.TestCase):
    def test_viewport_pixels(self):
        from pathapple.geometry import PANEL_SIZE, Pose, load_slides
        from pathapple.styles import STYLES
        record = json.loads((OUT / 'matches.json').read_text())
        slides = load_slides()
        cap = cv2.VideoCapture(str(OUT / 'bad-apple-pathology.mp4'))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.assertEqual(n, len(record['frames']))
        vx, vy = 400, 94          # layout A viewport origin
        W, H = PANEL_SIZE
        for t in np.linspace(0, n-1, 5).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
            ok, frame = cap.read()
            self.assertTrue(ok)
            m = record['frames'][t]['silhouette']
            p = Pose(**m['pose'])
            expected = STYLES['adaptive'](slides[p.slide], p, m['level'], W, H)
            got = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[vy:vy+H, vx:vx+W]
            # Compare away from the overlays (overview inset, scale bar, readout), at the
            # scale H.264 keeps: saturated fluorescence texture loses its finest detail.
            region = (slice(170, H-60), slice(10, W-240))
            got = cv2.GaussianBlur(got.astype(np.float32), (0, 0), 2)
            expected = cv2.GaussianBlur(expected.astype(np.float32), (0, 0), 2)

            def error(dx=0, dy=0):
                return np.abs(np.roll(got, (dy, dx), (0, 1))[region] - expected[region]).mean()
            err = error()
            self.assertLess(err, 6., f'frame {t}: viewport differs from the logged pose (mean error {err:.1f})')
            # And no small shift fits better, so the field is where the log says.
            for dx, dy in ((3, 0), (-3, 0), (0, 3), (0, -3)):
                self.assertLess(err, error(dx, dy), f'frame {t}: a {dx},{dy} pixel shift fits better')


if __name__ == '__main__':
    unittest.main()
