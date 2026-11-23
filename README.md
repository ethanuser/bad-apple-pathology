# Bad Apple!! but every frame is a histology slide

![Frame 330: the figure matched to the muscle wall of a slide of human sigmoid colon, shown in a slide-analysis workstation](docs/frame.jpg)

Every frame of Bad Apple!! is matched to the closest field of view in 111 H&E whole-slide images of normal human tissue, up to three from each of 38 tissues sampled by the GTEx project, from brain and pituitary to lung, colon, kidney, skin and tibial nerve. The slides are public, de-identified and CC BY 4.0, from the NCI Imaging Data Commons.

The video is laid out like a slide-analysis workstation:
- **Viewport.** The matched field in real stain colour: brightfield when the figure is dark, virtual fluorescence (hematoxylin blue, eosin magenta, on black) when it is light.
- **Image table.** The slide's pixel size and dimensions, the field size, rotation, stain channel and threshold, and the slide's own hematoxylin and eosin stain vectors.
- **Overview.** The whole slide with the field's footprint.
- **Reference.** The original Bad Apple!! frame.
- **Atlas.** A 3D body from the Human Reference Atlas with the sampled organ lit and the tissue block where GTEx registered that sample, beside the organ on its own.
- **Project.** All 38 tissues, with the current slide outlined and the similarity as a small trace.

There is also a 1080x1920 layout for phone feeds (`--layout vertical`), and two layouts after other slide viewers: a clinical slide tray (`--layout clinical`) and a full-screen scanner view (`--layout scanner`). See [the options](docs/previews/options.jpg).

## Quick start

Needs Python 3.11 to 3.13, FFmpeg and about 17 GB of disk: 5 GB of slides and anatomy, plus 12 GB of stain planes prepared on first use.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pathapple fetch     # downloads and verifies the slides, anatomy and media
.venv/bin/python -m pathapple run       # matches and renders a 7-second excerpt
```

The full song at 30 fps belongs on a CUDA GPU, then renders locally:

```sh
scripts/setup-env.sh && scripts/gpu-run.sh 30     # on the GPU machine
.venv/bin/python -m pathapple render --output output/full-song
.venv/bin/python -m pathapple render --output output/full-song --layout vertical
```

On the 7-second excerpt, mean similarity is 0.760. In the metrics ikuto32's Earth and Moon edition reports, NCC at 32 × 24 has median 0.895 and 10th percentile 0.861 (Moon 0.896 and 0.790), and mean foreground and background IoU is 0.881 (Earth about 0.80 to 0.83). The [MRI and CT edition](https://github.com/ethanuser/bad-apple-medical-scan) scores 0.743 similarity on the same seven seconds.

## More

- [Technical notes](docs/TECHNICAL.md): stain separation, search, scoring, display, anatomy, content rules, GPU runs and tests
- [Credits and licences](CREDITS.md)
