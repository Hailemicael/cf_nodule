"""Comparison + ablation figure (no in-image header; short panel labels only)."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("runs/benchmark.json")); T = R["_comparison_table"]
def au(n): return T[n]["auroc_mean"], T[n]["auroc_std"]
def iv(n): return T[n].get("auroc_intervention_mean")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))

comp = [("concept_only_trueMLP", "concept-only"), ("blackbox", "black-box"),
        ("cbm_joint", "joint CBM"), ("cbm_sequential", "sequential"),
        ("indep_wconcept_1.0", "indep. CBM\n(ours)")]
xs = np.arange(len(comp)); means = [au(n)[0] for n, _ in comp]; stds = [au(n)[1] for n, _ in comp]
cols = ["tab:blue", "0.6", "tab:orange", "tab:red", "tab:green"]
ax[0].bar(xs, means, yerr=stds, capsize=5, color=cols, alpha=0.9)
for i, (n, _) in enumerate(comp):
    ax[0].text(i, means[i] + stds[i] + 0.012, f"{means[i]:.3f}", ha="center", fontsize=9)
    v = iv(n)
    if v is not None:
        ax[0].scatter(i, v, color="black", marker="D", s=52, zorder=5)
ax[0].axhline(0.5, color="crimson", ls=":", lw=1)
ax[0].set_xticks(xs); ax[0].set_xticklabels([l for _, l in comp], fontsize=9)
ax[0].set_ylabel("malignancy AUROC"); ax[0].set_ylim(0.3, 1.0)
ax[0].set_title("comparison", fontsize=11)

groups = [("indep_head_linear", "linear"), ("cbm_independent_MAIN", "mlp32"),
          ("indep_head_mlp64", "mlp64"), ("indep_concepts_clin3", "clin-3"),
          ("indep_wconcept_0.25", "$w_c$0.25"), ("indep_wconcept_1.0", "$w_c$1.0")]
xs2 = np.arange(len(groups)); m2 = [au(n)[0] for n, _ in groups]; s2 = [au(n)[1] for n, _ in groups]
ax[1].bar(xs2, m2, yerr=s2, capsize=4, color="tab:green", alpha=0.85)
ax[1].scatter(xs2, [iv(n) for n, _ in groups], color="black", marker="D", s=44, zorder=5)
for i, v in enumerate(m2):
    ax[1].text(i, v + s2[i] + 0.012, f"{v:.3f}", ha="center", fontsize=8)
ax[1].axhline(T["blackbox"]["auroc_mean"], color="0.4", ls="--", lw=1, label="black-box")
ax[1].set_xticks(xs2); ax[1].set_xticklabels([l for _, l in groups], fontsize=9)
ax[1].set_ylim(0.3, 1.0); ax[1].set_ylabel("malignancy AUROC")
ax[1].set_title("ablations", fontsize=11); ax[1].legend(loc="lower right", fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig("runs/benchmark_fig.png", dpi=150, bbox_inches="tight")
print("[ok] benchmark_fig.png")
