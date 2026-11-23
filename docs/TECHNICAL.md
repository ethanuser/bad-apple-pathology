# Technical notes

## Inputs

`sources.json` pins every slide as two pyramid levels of a GTEx DICOM whole-slide image in the NCI Imaging Data Commons public bucket: the ~2 µm level (scanned at 20x, 0.494 µm per pixel, then reduced 4x by the scanner) for matching and display, and the ~8 µm level as the overview. Each file is fetched once over HTTPS and checked against its SHA-256. `scripts/select_slides.py` built the list: for each of GTEx's 40 coded anatomic sites it draws six candidate slides with a fixed seed and keeps the three with the most tissue on the glass (117 slides: a few sites had fewer usable candidates). Only the public specimen ID, the SNOMED-coded site and image geometry are recorded; donor age, sex and the free-text pathology notes in the DICOM headers are never read into the run.

`data/hra/` holds the Human Reference Atlas scene (organ placements and GTEx tissue blocks) as returned by the CCF API, kept in the repository because that API is not versioned, and `data/hra/meshes.json` pins the 3D organ meshes by hash. The Material Symbols font is pinned in `sources.json`.

## Stain planes

Each slide's 2 µm level is halved to about 4 µm, its glass colour is estimated from the brightest tenth of pixels, and optical density is separated into hematoxylin and eosin with stain vectors estimated per slide (Macenko et al. 2009). The textbook Ruifrok vectors do not fit GTEx's PAXgene-fixed, paler staining: they put the cerebellar granular layer in the eosin channel. Three matching channels are stored as transmitted brightness, so glass is 255 and dense stain is dark:

- **density**: total optical density, which outlines tissue against glass, lumens, fat and alveoli;
- **hematoxylin**: nuclei-dense regions such as lymphoid follicles, glands and the granular layer;
- **eosin**: cytoplasm, muscle and collagen.

Each channel is scaled to its 99.5th percentile within tissue. Tissue must be both dense and coloured, so grey dust, debris and pen marks on bare glass are not tissue. A seven-level pyramid (4 to 256 µm) of all three channels is cached beside the slide as memory-mapped `.npy` files, so a bank of 117 slides needs little RAM.

## Matching

1. **Coarse retrieval (GPU).** For every slide, four magnification bands (0.7–1.5, 1.5–3.2, 3.2–6.8 and 6.8–14 mm wide) and three channels, the search proposes fields centred on or near tissue with random width, angle and mirror, rejecting any whose footprint leaves the scanned area. Each band samples the pyramid level at about half its thumbnail pitch, so magnification never aliases. Fields are scored as 32 × 24 thumbnails against every target frame, mirrored and inverted: grayscale NCC (the baseline) and a contour score combining soft-thresholded correlation, gradient agreement and regional NCC in a 4 × 3 grid at a random level. The best 48 poses per target are kept for each score. Retrieval is saved, so an interrupted refinement resumes without it.
2. **Refinement (CPU).** Three rounds of joint perturbation of position, zoom, angle and *edge smoothing*, scored at 48 × 36 by balanced IoU, gradient agreement, regional NCC and thin detail. Edge smoothing is the Gaussian sigma of the silhouette field as a fraction of the field width (0.2–4.5%), so the same shape holds at any magnification and a field can range from cell-level outlines to smooth blobs. Each frame is also seeded with its predecessor's pose and that pose moved by the change in silhouette centroid and area.
3. **Selection.** Thresholds are fitted from class histograms, and the four neighbouring levels are tried on the full score. A dynamic program chooses the sequence with the highest total score less a small cost (up to 0.038) for changing slide, channel, mirror or polarity, or for jumps in position, zoom, angle and smoothing. A selection-only cost of up to 0.08 applies to fields with less than 30% stained tissue, so empty glass with a speck of dust cannot stand in for a silhouette.

**Similarity** is measured on the black and white silhouette at 96 × 72: 28% balanced IoU, 27% boundary F1 with a two-pixel tolerance, 20% thin detail, 12.5% positive grayscale NCC and 12.5% positive regional NCC. These are the same metrics as the MRI and CT project, so the numbers compare directly.

## Display

The viewport shows the real colour field from the 2 µm level (or its area-reduced halvings for wide fields), toned by one threshold for the whole field. The default *adaptive* style picks the illumination that keeps the match physically natural:

- black figure on white: **brightfield**, stain kept saturated on the dark side and faded toward clean glass on the other;
- white figure on black: **virtual fluorescence**, hematoxylin re-coloured blue and eosin magenta as emission on black, so the stained side glows.

`--style` also offers `brightfield`, `soft`, `fluorescence` and `duotone`.

## Layouts

`--layout workstation` (default) follows QuPath: menu and icon toolbar with the magnification readout, an Image table with the slide's pixel size, dimensions, field, rotation, channel, threshold, edge smoothing and its own stain vectors, the overview inset with the field's box, scale bar and a cursor readout of the field centre in micrometres and its colour. The left pane carries the source frame and the anatomy; under the viewer, the project's slides with the current one outlined and the similarity as a small trace. `clinical` follows Aperio WebViewer DX (slide tray, label, thumbnail heat map of where the video has looked, colour-coded zoom, rotation dial, magnifier) and `scanner` follows Hamamatsu NDP.view2 (full-bleed field with floating tool windows).

## Anatomy

The atlas renders the HRA 3D reference organs with OpenGL (moderngl): a translucent skin shell lit at its silhouette, context organs, the sampled organ or named sub-structure (for example the cerebellum, transverse colon or left ventricle) in the accent colour, and the sampling site as a 3D block. Seventeen GTEx tissues have blocks that GTEx registered into these organs, which are drawn at their registered size, position and orientation. Others use the named HRA structure. Tissues from organs the HRA does not model (stomach, oesophagus, adrenal gland, thyroid gland, testis, pituitary, and the lower-leg muscle, nerve, artery and fat sampled beside the lower-leg skin) are placed from neighbouring structures and drawn with a dashed ring.

GTEx's two colon registrations carry swapped names: the block registered as *Colon_Sigmoid* lies in, and is annotated as, the transverse colon (UBERON:0001157), and vice versa. Colon blocks are therefore chosen by their anatomical annotation. The body's sex follows the tissue (male for prostate, testis and, where only a male block exists, sigmoid colon; otherwise female), not the donor. Rendered views are cached on disk so the forked frame renderers never need an OpenGL context.

## Commands

```sh
.venv/bin/python -m pathapple fetch
.venv/bin/python -m pathapple run                       # 7-second excerpt: match and render
.venv/bin/python -m pathapple match --start 0 --seconds 232 --output output/full-song
.venv/bin/python -m pathapple render --output output/full-song --layout workstation
```

`run` and `match` accept `--start`, `--seconds`, `--fps`, `--seed`, `--samples`, `--coarse-width`, `--workers` and `--device`. `render` accepts `--layout`, `--style` and `--to`. Rendering runs in a fresh process because Metal, used by the search on a Mac, cannot be forked.

## Verification

`.venv/bin/python -m unittest discover -s tests -v`

- `test_geometry`: the silhouette field and the displayed colour crop sample the same tissue (an early version assumed 4.0 µm planes where the slides give 3.95 µm, shifting the colour image by up to a quarter of a field), and the logged footprint is the mapped field corners.
- `test_anatomy`: every GTEx tissue has a site inside the body, registered tissues use GTEx's blocks, and the colon blocks sit in the colon segment they are annotated as.
- `test_privacy`: no donor fields or free text in the pinned sources or a saved run.
- `test_rendered_provenance`: decodes the video and re-renders sampled viewports from `matches.json` and the pinned slides, within a lossy-codec tolerance.
