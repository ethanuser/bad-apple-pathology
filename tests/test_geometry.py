"""The silhouette field (stain planes) and the displayed colour crop must sample the
same tissue. An early version assumed 4 µm planes where the slides give 3.95 µm, which
shifted the colour image off the matched shape by up to a quarter of a field."""
import unittest

import numpy as np

from pathapple.geometry import Pose, colour_crop, field, footprint, inside, maps, values
from pathapple.inputs import ASSETS, SOURCES


def ready():
    return all((ASSETS / s['key'] / 'fine.dcm').exists() for s in SOURCES['slides'][:2])


@unittest.skipUnless(ready(), 'slides not fetched')
class FieldAlignment(unittest.TestCase):
    def test_field_matches_colour(self):
        from pathapple.geometry import load_slides
        from pathapple.slides import optical_density
        slides = load_slides()
        rng = np.random.default_rng(0)
        for si in (0, len(slides)//2):
            s = slides[si]
            ys, xs = np.nonzero(s.tissue_mask)
            for _ in range(4):
                k = rng.integers(len(xs))
                p = Pose(si, 0, (xs[k]+.5)*4*s.base_um, (ys[k]+.5)*4*s.base_um, float(rng.uniform(800, 3000)),
                         float(rng.uniform(-180, 180)), bool(rng.integers(2)), False, 0.)
                if not inside([values(p)], s.size_um)[0]:
                    continue
                f = field(s, p, 96, 72)
                rgb = colour_crop(s, p, 96, 72)
                density = optical_density(rgb, s.glass).sum(axis=2)
                r = np.corrcoef(f.ravel(), -density.ravel())[0, 1]
                self.assertGreater(r, .85, f'{s.key}: field and colour disagree (r={r:.2f})')

    def test_footprint_is_the_mapped_corners(self):
        from pathapple.geometry import load_slides
        s = load_slides()[0]
        p = Pose(0, 0, s.size_um[0]/2, s.size_um[1]/2, 2000., 30., True, False)
        corners = np.array(footprint(s, p, 1.))
        x, y = maps([values(p)], 1., 0, 0, corners=True)
        self.assertTrue(np.allclose(corners, np.stack([x.ravel()+.5, y.ravel()+.5], 1)))
        # Mirroring swaps left and right corners; the field's width is the edge length.
        self.assertAlmostEqual(np.linalg.norm(corners[1]-corners[0]), 2000., places=2)
        self.assertAlmostEqual(np.linalg.norm(corners[3]-corners[0]), 1500., places=2)


if __name__ == '__main__':
    unittest.main()
