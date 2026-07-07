"""Qualitative example nodules, clean (no in-image header; caption carries the text)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("data/cache/lidc_patches.npz", allow_pickle=True)
img, mask, con, y = d["image"], d["mask"], d["concept"], d["y"]
SP, LO, MG = 5, 4, 3  # spiculation, lobulation, margin
rng = np.random.default_rng(3)
idx = list(rng.choice(np.where(y == 0)[0], 3, replace=False)) + \
      list(rng.choice(np.where(y == 1)[0], 3, replace=False))

fig, ax = plt.subplots(1, 6, figsize=(13, 2.7))
for a, k in zip(ax, idx):
    a.imshow(img[k, 0], cmap="gray")
    a.contour(mask[k, 0], levels=[0.5], colors="lime", linewidths=1.0)
    a.set_title("malignant" if y[k] == 1 else "benign", fontsize=11,
                color=("firebrick" if y[k] == 1 else "seagreen"))
    a.set_xlabel(f"sp {con[k,SP]:.2f}  lo {con[k,LO]:.2f}  mg {con[k,MG]:.2f}", fontsize=8)
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig("runs/examples.png", dpi=150, bbox_inches="tight")
print("[ok] examples.png")
