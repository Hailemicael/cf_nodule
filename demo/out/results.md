# CF-Nodule demo results (synthetic)

Proof-of-concept of the BIBM method on synthetic nodules with a planted dataset shortcut. Numbers are illustrative of the *mechanism*, not clinical results.

| Metric | Vanilla | CFA (ours) |
|---|---|---|
| malignancy_AUROC | 0.732 | 0.919 |
| faithfulness | 0.838 | 0.911 |
| concept_alignment | 0.104 | 0.132 |
| CFS | 0.077 | 0.121 |

Key: CFA improves intervention-faithfulness and concept-alignment, and the combined CFS enables better selective referral (see fig4).
