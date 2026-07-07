# CF-Nodule: Faithful, Interpretable-by-Design Lung-Nodule Malignancy Assessment

Concept-bottleneck models for lung-nodule malignancy on **LIDC-IDRI** that are
**verifiably faithful** (the decision flows through clinical concepts and can be
probed by intervention), **scanner-robust**, and **as accurate as a black box** —
together with a reproducible evaluation protocol that exposes common false positives
in attribution-faithfulness studies.

> The data pipeline uses the **public TCIA REST API** + `pylidc`, so there is **no
> gated 125 GB download**. Everything runs on a single consumer GPU.

## Key results (full LIDC, 983 patients / 1185 nodules, patient-level 5-fold CV)

| Model | AUROC | Concept intervention |
|---|---|---|
| Black-box CNN | 0.908 | n/a |
| Independent CBM (strict scalar bottleneck) | 0.866 | 0.855 |
| Concept-Embedding CBM (CEM) | 0.903 | 0.904 |
| **CEM + 2.5D + ensemble (best)** | **0.920** | **0.920** |
| Black-box + 2.5D + ensemble | 0.919 | n/a |

- The **faithful** model matches/edges the black box (0.920 vs 0.919) while remaining
  fully interventionable — interpretability at no accuracy cost.
- **Cross-scanner** leave-one-manufacturer-out: mean AUROC **0.877** (GE/Siemens/Philips/Toshiba).
- **Selective referral:** accuracy rises from ~0.70 to ~0.75 deferring the least-confident 20%.
- **Reproducibility finding:** a post-hoc Grad-CAM faithfulness regularizer (CFA)
  shows single-split "significance" (p<0.005) that **reverses across seeds and vanishes
  under 5-fold CV**; naively pooling per-nodule scores manufactures p=0.016 that the
  per-fold test rejects (p=0.86). Multi-seed, patient-level CV is necessary.

*Numbers are from 2D single/2.5D patches under strict patient-level CV; ~0.96-level
numbers in the literature typically require 3D models or leak-prone nodule-level splits.*

## Install
```bash
pip install -r requirements.txt
# PyTorch: install the build matching your GPU/CUDA from pytorch.org.
# (e.g., Blackwell/RTX 50-series needs the cu128 wheel: torch 2.7+.)
```

## Data (no gated download)
LIDC-IDRI is a public TCIA collection, and `pylidc` ships the radiologist annotations
(malignancy + 8 semantic concepts). We only fetch the DICOM pixels for a subset of
patients over the public API:

```bash
python -m src.data.fetch_lidc_api --patients 300 --out data/LIDC-IDRI   # public TCIA API, no login
python -m src.data.download_lidc --set-path data/LIDC-IDRI --check       # writes pylidc config
python -m src.data.lidc_dataset --build-cache --config configs/default.yaml
```
Notes (handled in code): on Windows pylidc's config file is `~/pylidc.conf`; pylidc
(<=0.2.3) needs `configparser.SafeConfigParser` and `np.int` shims under Python 3.12 /
NumPy 2.x — applied automatically in `src/data/lidc_dataset.py`.

## Quick start (no data, no GPU)
```bash
python demo/cf_demo.py --outdir demo/out          # synthetic mechanism demo
```

## Models
A shared U-Net encoder feeds three heads (segmentation, concepts, malignancy). The
malignancy pathway is the object of study:

- **black-box** — malignancy from image features (standard CNN).
- **joint CBM** — malignancy from predicted concepts, trained end-to-end (accurate but *leaky*).
- **sequential / independent CBM** — independent trains the concept→malignancy head on
  ground-truth concepts (faithful by construction).
- **CEM** — concept-embedding bottleneck; recovers black-box accuracy while staying interventionable.
- **CFA** — post-hoc Grad-CAM faithfulness+alignment regularizer (baseline that does not reproduce).

## Reproducing the experiments
| Script | What it does |
|---|---|
| `kfold.py` | patient-level 5-fold CV, paired statistics + concept-intervention test |
| `benchmark.py` | full method comparison + ablations (head, concept set, loss weight) |
| `cbm_kfold.py`, `cbm_independent.py` | joint vs independent CBM, leakage detection |
| `cem.py` | concept-embedding model (5-fold) |
| `build_cache_25d.py`, `ensemble_25d.py` | 2.5D multi-slice cache + ensemble (best config) |
| `final_main.py` | full-LIDC faithful model + risk-coverage + calibration |
| `blackbox_full.py` | black-box baseline on full LIDC |
| `cross_scanner.py` | leave-one-manufacturer-out generalization |
| `multiseed.py` | multi-seed validation of the CFA regularizer (reproducibility study) |
| `transformer_cbm.py` | pretrained ViT encoder ablation |
| `make_*.py`, `extract_examples.py` | figures |

Results are written to `runs/` (git-ignored). See `runs/RESULTS_real.md` after running.

## Evaluation protocol
Patient-level k-fold CV; mean ± s.d. with paired per-fold tests; multi-seed validation;
and a **concept-intervention test** (feed ground-truth concepts, re-measure AUROC) that
distinguishes genuinely concept-grounded models from accurate-but-leaky ones.

## License
MIT — see [LICENSE](LICENSE).
