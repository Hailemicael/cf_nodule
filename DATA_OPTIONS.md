# Getting the data without drowning in 125 GB

You do **not** need the full raw LIDC-IDRI DICOM archive. CF-Nodule only consumes
small 64×64 patches + labels, so the practical paths are:

## Recommended: run on Kaggle / Colab (no local storage)
Free GPU, datasets already hosted. Use `notebooks/cf_nodule_colab.ipynb`.

### Route A — concept-complete (best for the paper)
The CFA method needs the LIDC **semantic concepts** (spiculation, lobulation,
margin, …). These live in the LIDC XML annotations and are read by `pylidc`,
which needs the **DICOM** version.

1. On Kaggle: *Add Data* → search **"LIDC-IDRI"** → attach a DICOM dataset
   (it mounts read-only at `/kaggle/input/...`, nothing downloaded to you).
2. The notebook configures `~/.pylidcrc` to point at that path and runs
   `src.data.lidc_dataset --build-cache`, which extracts patches + masks +
   all 7 concept ratings + malignancy into a ~few-hundred-MB `.npz`.
3. Train + evaluate from the cache.

### Route B — fast images + key concepts
- Images/masks from a hosted **LUNA16** dataset (888 scans, mhd/raw):
  - https://www.kaggle.com/datasets/avc0706/luna16
  - https://www.kaggle.com/datasets/eliasmarcon/luna-16
  - https://www.kaggle.com/datasets/fanbyprinciple/luna-lung-cancer-dataset
- Concepts (spiculation/lobulation) from **CIRDataset** (LIDC N=883, masks +
  spiculation/lobulation): https://github.com/choasma/CIRDataset
  (paper: https://arxiv.org/abs/2206.14903)

This route gives you the two most clinically decisive concepts, which is enough
for the headline CFA result.

## If you insist on local (you have 30–150 GB)
- **Subset download:** edit the NBIA `.tcia` manifest to ~100–200 patients
  (~15–25 GB). Keep the patient-level split. Enough for a conference result.
- **Process-and-delete:** download in batches → `build-cache` → delete the raw
  DICOM. Peak disk stays small; the `.npz` cache is tiny.
- **LUNA16 only:** ~size of 10 subsets; pair with CIRDataset for concepts.

## Handy preprocessing tools (read the XML for you)
- pylidc (what our loader uses): https://pylidc.github.io/
- jaeho3690/LIDC-IDRI-Preprocessing (Image/Mask/Meta + malignancy):
  https://github.com/jaeho3690/LIDC-IDRI-Preprocessing
- qiuliwang/LIDC-IDRI-Toolbox-python (nodule characteristics CSV incl. spiculation):
  https://github.com/qiuliwang/LIDC-IDRI-Toolbox-python

## TL;DR
Use Kaggle, attach a hosted LIDC DICOM dataset, let the notebook build a small
cache with `pylidc`. You never store 125 GB anywhere.
