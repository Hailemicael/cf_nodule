"""CBM figure (no in-image header; short panel labels). Numbers consistent with paper."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

bb   = [0.905, 0.909, 0.870, 0.826, 0.898]
jcbm = [0.917, 0.901, 0.874, 0.835, 0.893]
icbm = [0.908, 0.911, 0.905, 0.857, 0.892]
jintv, iintv = 0.41, 0.855
imp = {"lobulation": 0.262, "spiculation": 0.229, "margin": 0.213,
       "subtlety": 0.082, "texture": 0.082, "sphericity": 0.075, "calcification": 0.057}

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

groups = [("black-box", bb, "0.6"), ("joint CBM", jcbm, "tab:orange"),
          ("indep. CBM\n(ours)", icbm, "tab:green")]
for i, (lab, arr, col) in enumerate(groups):
    ax[0].bar(i, np.mean(arr), color=col, alpha=0.9)
    ax[0].scatter([i] * len(arr), arr, color="black", s=16, zorder=3)
    ax[0].text(i, np.mean(arr) + 0.013, f"{np.mean(arr):.3f}", ha="center", fontsize=9)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([g[0] for g in groups], fontsize=9)
ax[0].set_ylim(0.6, 1.0); ax[0].set_ylabel("malignancy AUROC"); ax[0].set_title("accuracy", fontsize=11)

ax[1].bar([0, 1], [jintv, iintv], color=["tab:orange", "tab:green"])
ax[1].axhline(0.5, color="crimson", ls="--", lw=1, label="chance")
for i, v in enumerate([jintv, iintv]):
    ax[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["joint CBM\n(leaky)", "indep. CBM\n(faithful)"], fontsize=9)
ax[1].set_ylim(0, 1.0); ax[1].set_ylabel("AUROC with true concepts")
ax[1].set_title("concept intervention", fontsize=11); ax[1].legend(loc="upper left", frameon=False)

items = sorted(imp.items(), key=lambda kv: kv[1]); names = [k for k, _ in items]; vals = [v for _, v in items]
colors = ["tab:green" if n in ("spiculation", "lobulation", "margin") else "0.6" for n in names]
ax[2].barh(range(len(names)), vals, color=colors)
ax[2].set_yticks(range(len(names))); ax[2].set_yticklabels(names, fontsize=9)
ax[2].set_xlabel(r"relative importance $|\partial\,\mathrm{logit}/\partial c|$")
ax[2].set_title("concept importance", fontsize=11)

fig.tight_layout()
fig.savefig("runs/cbm_final.png", dpi=150, bbox_inches="tight")
print("[ok] cbm_final.png")
