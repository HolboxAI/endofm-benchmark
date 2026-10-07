# Endo-FM benchmark on Kvasir-Capsule

Benchmarks the [Endo-FM](https://github.com/med-air/Endo-FM) video foundation
model (ViT-B/16 TimeSformer, self-supervised pretrained on endoscopic video)
as a frozen feature extractor on the
[Kvasir-Capsule](https://github.com/simula/kvasir-capsule) video capsule
endoscopy dataset.

Run: **`benchmark_endofm_kvasir_capsule.ipynb`**

## Why Kvasir-Capsule instead of Endo-FM's own PolypDiag benchmark

Endo-FM's paper reports downstream numbers on PolypDiag / CVC-12k / KUMC, but
the preprocessed datasets *and* fine-tuned checkpoints for all three are
hosted on the authors' personal CUHK SharePoint links, which are dead
(`404 FILE NOT FOUND`, confirmed 2026-09-16 for all three datasets and two of
the three checkpoints — a live upstream issue, `med-air/Endo-FM#33`, has
someone else asking the authors for a fresh copy). Only the general
pretrained backbone (`endo_fm.pth`, Google Drive) is still reachable. So this
benchmarks the working backbone against a dataset that's still live, rather
than reproducing a paper number that currently can't be reproduced by anyone.

## Directory layout

```
endofm-benchmark/
  Endo-FM/                                 official repo (git clone, model code only)
  checkpoints/
    endo_fm.pth                            pretrained ViT-B/16 backbone (Google Drive)
  data/
    kvasir-capsule-labeled-videos.zip       43 labeled procedure videos (32.2GB)
    kvasir-capsule-unlabeled-videos.zip     74 unlabeled videos (58.2GB) -- staged, not used yet
    kvasir-capsule-labeled-videos/
      raw/                                  extracted video files
      extracted_frames/<video_id>/<frame>.jpg   frames named in the official split, decoded on demand
    official_splits/split_0.csv, split_1.csv    official 2-fold evaluation protocol
  results/
    benchmark_results.json, confusion_matrix.png
  benchmark_endofm_kvasir_capsule.ipynb
  build_notebook.py                        regenerates the notebook from source (not part of the analysis)
```

## What the notebook does

1. Verifies/downloads the Endo-FM backbone + Kvasir-Capsule labeled videos.
2. Extracts exactly the frames named in Kvasir-Capsule's official 2-fold
   split directly from the raw videos (single sequential decode pass per
   video).
3. Linear-probes frozen Endo-FM CLS-token features (each frame replicated
   across the model's 8-frame temporal window) against the official 2-fold
   protocol (train fold 0/test fold 1 and vice versa).
4. Repeats the same pipeline with a randomly-initialized backbone as a
   control, to isolate what the pretraining is actually buying.
5. Reports accuracy, macro-F1, per-class breakdown, confusion matrix.

## What it deliberately doesn't do

The unlabeled videos (58.2GB) are **deprioritized for now, at the user's
request** (a download of the unlabeled zip was started, then killed within
seconds via a watcher process once it started writing, to avoid wasting
bandwidth ahead of the labeled-video benchmark). They match the "DAPT:
continued self-supervised pretraining on unlabeled video" step in
`neurosurgery_notegen_architecture.md` (Layer 2), which is a separate, much
larger undertaking (Endo-FM's own multi-crop DINO-style SSL loop) out of
scope for a benchmark notebook. Re-download on request when that's next.

## Running on Google Colab

The notebook auto-detects Colab in Section 0 (`try: import google.colab`) and
adapts itself -- no manual path edits needed.

Only the notebook itself needs to go to Drive -- everything else (`Endo-FM/`
repo, `endo_fm.pth` checkpoint, video zips, official split CSVs) is
downloaded/re-extracted by the notebook on first run, straight into Drive:

1. Upload `benchmark_endofm_kvasir_capsule.ipynb` to Google Drive, e.g. as
   `My Drive/endofm-benchmark/benchmark_endofm_kvasir_capsule.ipynb`.
2. Open it from Drive: colab.research.google.com -> File -> Open notebook ->
   Google Drive, or right-click the file in Drive -> Open with -> Google
   Colaboratory.
3. **Runtime -> Change runtime type -> T4 GPU**, then **Runtime -> Run all**.
4. Section 0 mounts Drive, points every path at
   `/content/drive/MyDrive/endofm-benchmark`, and installs the packages Colab
   doesn't ship (`einops`, `timm`, `gdown`, `opencv-python-headless`). Section 1
   clones `Endo-FM` and downloads the checkpoint; Section 2 downloads +
   extracts the Kvasir-Capsule labeled videos and official split CSVs -- all
   skipped automatically on any later run once they're already on Drive.

**Storage**: the first run downloads/creates ~40GB under that Drive folder
(32.2GB labeled-video zip + extracted frames + 2.3GB checkpoint). Because it
lands on Drive rather than Colab's ephemeral local disk, you only pay that
download/extraction/feature-extraction cost once -- it survives runtime
disconnects and later sessions reuse it. `build_notebook.py` and
`extract_frames.py` are local dev tooling (regenerate the notebook / a
standalone frame-extraction script) and aren't needed on Colab.

## Environment (local)

The system Python (3.14) has no CUDA-enabled PyTorch wheel available yet, and
this model is a video transformer (each still frame becomes an 8-frame clip),
so CPU throughput is impractical: 0.44 images/sec measured, i.e. ~30 hours for
the full official split. Instead, a dedicated venv
**`.venv-gpu`** (Python 3.9 + `torch==2.5.1+cu121`) targets the local GTX 1650
(4GB), giving ~3.5 images/sec (~3.75h for the full split) -- registered as the
Jupyter kernel **`endofm-gpu`** ("Python (endofm-gpu, CUDA)"). The notebook's
kernelspec is generic (`python3`) for Colab compatibility, so select the
`endofm-gpu` kernel manually from the Jupyter kernel picker when running
locally. Recreate the venv with:

```
py -3.9 -m venv .venv-gpu
.venv-gpu\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
.venv-gpu\Scripts\python -m pip install einops timm opencv-python scikit-learn matplotlib tqdm pandas requests gdown ipykernel nbformat nbclient
.venv-gpu\Scripts\python -m ipykernel install --user --name endofm-gpu --display-name "Python (endofm-gpu, CUDA)"
```

## Provenance / citations

- Wang et al., *Foundation Model for Endoscopy Video Analysis via
  Large-scale Self-supervised Pre-train*, MICCAI 2023.
  [arXiv:2306.16741](https://arxiv.org/abs/2306.16741) ·
  [GitHub](https://github.com/med-air/Endo-FM)
- Smedsrud et al., *Kvasir-Capsule, a video capsule endoscopy dataset*,
  Scientific Data 8, 142 (2021).
  [doi:10.1038/s41597-021-00920-z](https://www.nature.com/articles/s41597-021-00920-z) ·
  [GitHub](https://github.com/simula/kvasir-capsule) ·
  [dataset](https://datasets.simula.no/kvasir-capsule/)
