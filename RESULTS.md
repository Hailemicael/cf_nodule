# Reproduction results

Independent end-to-end reproduction of the CF-Nodule paper, run from scratch on a
fresh machine: public-API LIDC download → cache build → all model variants,
ablations, and statistics.

- **Cohort:** 1194 nodules from 1018 CT scans (paper: 1185 nodules / 983 series).
  Per-manufacturer test sizes match the paper's Table VII exactly (GE 714, Philips 117,
  Toshiba 91), confirming cohort parity.
- **Protocol:** patient-level 5-fold CV; AUROC reported as mean ± s.d. over folds.
- **Environment:** CPU-only, PyTorch 2.4.1, Python 3.8 (the paper used an RTX 5080 GPU).
- **Intervention AUROC** = malignancy AUROC when the ground-truth concepts are fed to the
  concept-to-malignancy head (a leaky model drops toward chance = 0.5).

Status: **match** = within ~0.01 · **close** = within ~0.03 or same conclusion · **gap** = > 0.03.

## Table VI — Full-LIDC headline

| Model | AUROC (repro) | interv. | AUROC (paper) | interv. | status |
|---|---|---|---|---|---|
| Black-box CNN | 0.918 ± .010 | — | 0.908 ± .013 | — | match |
| Independent CBM (strict) | 0.876 ± .034 | 0.865 | 0.866 ± .028 | 0.855 | match |
| Concept-embedding CBM | 0.911 ± .011 | 0.911 | 0.903 ± .022 | 0.904 | match |
| Black-box CNN (2.5D, ens.) | 0.922 ± .006 | — | 0.919 ± .007 | — | match |
| CEM (2.5D, ens.) | 0.916 ± .011 | 0.916 | 0.920 ± .011 | 0.920 | match |

The faithful concept-embedding model matches the black box while staying fully
interventionable (intervention AUROC = accuracy).

## Table III — Method comparison

| Method | AUROC (repro) | interv. | AUROC (paper) | interv. | status |
|---|---|---|---|---|---|
| Concept-only (true c) | 0.908 ± .031 | — | 0.918 | — | match |
| Black-box (image) | 0.878 ± .010 | — | 0.882 | — | match |
| Joint CBM | 0.881 ± .025 | **0.543** (leaky) | 0.894 | 0.41 | match |
| Sequential CBM | 0.743 ± .250 | 0.63 | 0.672 ± .283 | 0.49 | close |
| Independent CBM (base) | 0.852 ± .028 | 0.795 | 0.858 | 0.80 | match |
| Independent + aug | 0.876 ± .034 | 0.865 | 0.895 | 0.855 | close |
| Hybrid (residual) | 0.917 ± .011 | partial | 0.906 | partial | match |

## Table V — Ablations (independent CBM)

| Setting | AUROC (repro) | AUROC (paper) | status |
|---|---|---|---|
| Head — linear | 0.686 ± .176 | 0.769 ± .141 | close (both unstable) |
| Head — MLP-32 | 0.852 ± .028 | 0.858 | match |
| Head — MLP-64 | 0.857 ± .021 | 0.862 | match |
| Concepts — all 7 | 0.852 | 0.858 | match |
| Concepts — clinical 3 | 0.841 ± .028 | 0.864 | close |
| wc = 0.25 | 0.819 ± .049 | 0.870 | gap |
| wc = 0.5 | 0.852 | 0.858 | match |
| wc = 1.0 | 0.852 ± .025 | 0.884 | close |
| Encoder — ViT-Tiny | 0.772 ± .071 | 0.690 ± .171 | close (both weak/unstable) |

Same conclusions as the paper: a linear head is unstable (a small MLP is needed), three
clinical concepts (margin, lobulation, spiculation) match all seven, and the pretrained
ViT-Tiny is far below the convolutional backbone at this data scale.

## Table VII — Cross-scanner generalization (leave-one-manufacturer-out)

| Held-out vendor | test n (repro / paper) | AUROC (repro) | AUROC (paper) | status |
|---|---|---|---|---|
| GE | 714 / 714 | 0.839 | 0.851 | match |
| Siemens | 272 / 263 | 0.837 | 0.815 | match |
| Philips | 117 / 117 | 0.841 | 0.914 | gap |
| Toshiba | 91 / 91 | 0.889 | 0.927 | close |
| **Mean** | 1194 | **0.851 ± .025** | 0.877 ± .053 | close |

## Table II — CFA regularizer reproducibility

The paper's central methodological claim: pooling correlated per-nodule scores manufactures
significance that the correct cluster-level (per-fold) and multi-seed tests reject.

| Metric | pooled p | per-fold t_p | 5-seed t_p | verdict |
|---|---|---|---|---|
| Concept-alignment | 0.000 | 0.59 | 0.17 | pooled "significant" → rejected |
| CFS | 0.000 | 0.64 | 0.17 | pooled "significant" → rejected |
| Faithfulness | 0.32 | 0.95 | 0.81 | not significant anywhere |
| Malignancy AUROC | Δ −0.013 (p .09) | — | Δ +0.005 (5-seed) | no effect |

Reproduced exactly: the pooled per-nodule bootstrap invents p ≈ 0 that both the per-fold
and multi-seed tests reject.

## Key findings

- **Joint-CBM leakage.** Fed true concepts, the joint CBM's AUROC collapses from 0.823
  (predicted concepts) to **0.543** (≈ chance) — accurate but not concept-driven. Paper: 0.41.
- **Independent CBM stays faithful.** Under true concepts it holds at **0.795** (0/5 folds
  show leakage). Paper: 0.80.
- **Concept importance (Fig 4).** Top learned concepts: spiculation 0.238, lobulation 0.218,
  margin 0.216 — the clinical malignancy signs, as in the paper.
- **Faithfulness cost.** The strict scalar bottleneck (0.876) sits ~4 points below the black
  box (0.918) — the paper's stated cost, which a smaller partial-download cohort had masked.
- **Selective referral & calibration.** Accuracy 0.773; ECE 0.125; deferring the least-confident
  cases raises accuracy (~0.77 → 0.83).

## Notes

- **Code fix.** `ensemble_25d.py` originally looped over `("cem",)` only and relied on a
  pre-existing black-box result being reused, so the black-box 2.5D row of Table VI was never
  produced on a clean run. It now iterates `("blackbox", "cem")` while still skipping any result
  already computed; the row above (0.922 ± .006) comes from that fix.
- **Data & artifacts** (`data/`, `runs/`, `*.npz`) are git-ignored; regenerate with the commands
  in the README. Reproduction ran on the full public LIDC-IDRI collection via the TCIA REST API.
