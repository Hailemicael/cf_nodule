"""Per-seed CFA deltas (bottleneck run, matches paper text). No in-image header."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("runs/multiseed_summary_bottleneck.json"))
per, summ = d["per_seed"], d["across_seed_delta"]
metrics = ["faithfulness", "concept_alignment", "concept_mass_in_band", "CFS"]
labels = ["faithfulness", "concept\nalignment", "concept\nmass", "CFS"]

fig, ax = plt.subplots(figsize=(8.4, 4.3))
for i, m in enumerate(metrics):
    vals = [r[m] for r in per]
    ax.scatter([i] * len(vals), vals, s=42, color="0.55", zorder=3)
    mean, std = summ[m]["mean_delta"], summ[m]["std"]
    ax.errorbar(i, mean, yerr=std, fmt="o", color="crimson", capsize=6, ms=8, zorder=4,
                label="mean $\\pm$ s.d." if i == 0 else None)
    ax.text(i + 0.13, mean, f"p={summ[m]['t_p']:.2f}", fontsize=9, va="center")
ax.axhline(0, color="black", ls="--", lw=1)
ax.set_xticks(range(len(metrics))); ax.set_xticklabels(labels)
ax.set_ylabel("paired difference  (CFA $-$ baseline)")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig("runs/multiseed_effects.png", dpi=150, bbox_inches="tight")
print("[ok] multiseed_effects.png")
