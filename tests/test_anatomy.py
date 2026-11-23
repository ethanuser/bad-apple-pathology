"""Every GTEx tissue gets a sampling site; registered tissues use GTEx's own blocks."""
import unittest

import numpy as np

from pathapple.anatomy import HRA, TISSUES
from pathapple.inputs import HRA_MESHES

GTEX_TISSUES = {
    'Adipose tissue', 'Adrenal gland', 'Aorta', 'Auricular appendage', 'Breast', 'Cerebellum', 'Cerebral cortex',
    'Coronary artery', 'Endocervix', 'Esophageal mucous membrane', 'Esophageal muscle', 'Exocervix',
    'Fallopian tube', 'Gastroesophageal junction', 'Heart', 'Ileum', 'Liver', 'Lung', 'Minor salivary gland',
    'Omentum', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate', 'Renal cortex', 'Renal medulla', 'Sigmoid colon',
    'Skeletal muscle', 'Skin of lower leg', 'Skin of suprapubic region', 'Spleen', 'Stomach', 'Testis', 'Thyroid',
    'Tibial artery', 'Tibial nerve', 'Transverse colon', 'Urinary bladder', 'Uterus', 'Vagina'}


class Mapping(unittest.TestCase):
    def test_all_tissues_mapped(self):
        self.assertEqual(GTEX_TISSUES - set(TISSUES), set())


@unittest.skipUnless(all((HRA.parent / m['path']).exists() for m in HRA_MESHES), 'anatomy not fetched')
class Sites(unittest.TestCase):
    def test_every_tissue_has_a_site_inside_the_body(self):
        from pathapple.anatomy import body, sex_for
        for tissue in sorted(GTEX_TISSUES):
            b = body(sex_for(tissue))
            point, corners, approx = b.site(tissue)
            self.assertIsNotNone(point, tissue)
            skin = b.points('skin')
            lo, hi = skin.min(axis=0), skin.max(axis=0)
            self.assertTrue(np.all(point > lo) and np.all(point < hi), tissue)
            spec = TISSUES[tissue]
            if (spec.get('block') or spec.get('uberon')) and not spec.get('depth'):
                self.assertIsNotNone(corners, f'{tissue} should use a GTEx registered block')
                self.assertFalse(approx, tissue)

    def test_colon_blocks_follow_their_annotation(self):
        from pathapple.anatomy import body
        for sex in ('female', 'male'):
            b = body(sex)
            for tissue, sub, other in (('Transverse colon', 'transverse_colon', 'sigmoid_colon'),
                                       ('Sigmoid colon', 'sigmoid_colon', 'transverse_colon')):
                point, corners, _ = b.site(tissue)
                if corners is None:
                    continue
                near = np.linalg.norm(b.points('large intestine', sub) - point, axis=1).min()
                far = np.linalg.norm(b.points('large intestine', other) - point, axis=1).min()
                self.assertLess(near, far, f'{tissue} block in the {sex} body sits nearer the {other}')


if __name__ == '__main__':
    unittest.main()
