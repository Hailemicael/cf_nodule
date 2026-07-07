# CF-Nodule demo results (synthetic)

Proof-of-concept of the BIBM method on synthetic nodules with a planted dataset shortcut. Numbers are illustrative of the *mechanism*, not clinical results.

| Metric | Vanilla | CFA (ours) |
|---|---|---|
| malignancy_AUROC | 0.732 | 0.918 |
| faithfulness | 0.825 | 0.757 |
| concept_alignment | 0.104 | 0.132 |
| CFS | 0.075 | 0.103 |

Key: CFA improves intervention-faithfulness and concept-alignment, and the combined CFS enables better selective referral (see fig4).
