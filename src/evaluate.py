"""
Evaluate trained CF-Nodule checkpoints with the faithfulness-first protocol and
PAIRED statistics (both models scored on the same test nodules), then produce the
BIBM result figures.

    python -m src.evaluate --config configs/default.yaml \
        --ckpt runs/cfnodule_cfa-on/best.pt --baseline runs/cfnodule_cfa-off/best.pt

Reports, per metric: each model's mean [95% CI], the PAIRED mean difference
(CFA - baseline) with bootstrap 95% CI and Wilcoxon p, and a paired AUROC test.
Paired tests remove between-nodule variance and are the correct, powerful
comparison for two models evaluated on identical cases.
"""
import os, argparse, yaml, json
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.models.cf_nodule import CFNodule
from src.data.lidc_dataset import make_datasets
from src import metrics as M


def load(ckpt, dev):
    s = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = s["cfg"]; n = len(cfg["data"]["concepts"])
    m = CFNodule(n_concepts=n, base=cfg["model"]["base_channels"], depth=cfg["model"]["depth"])
    m.load_state_dict(s["model"]); m.to(dev).eval()
    return m, cfg


def assess(model, te, dev, steps, n_th):
    """Per-nodule arrays (same order for every model -> paired)."""
    loader = DataLoader(te, batch_size=1)
    F_, CA, MASS, LES, probs, ys = [], [], [], [], [], []
    for img, mask, concept, region, y in loader:
        img = img.to(dev)
        cam = model.gradcam_malignancy(img).detach()
        ins, dele, _, _ = M.insertion_deletion(model, img, cam, steps=steps)
        F_.append(M.faithfulness_score(ins, dele))
        CA.append(M.auc_iou(cam, region, n_th))            # threshold-free IoU (strict)
        LES.append(M.auc_iou(cam, mask, n_th))
        c = cam.squeeze().cpu().numpy(); r = region.squeeze().cpu().numpy()
        MASS.append(float((c * r).sum() / (c.sum() + 1e-6)))  # mass-in-band (loss target)
        probs.append(torch.sigmoid(model(img)["malignancy"]).item()); ys.append(int(y))
    arr = lambda x: np.array(x, float)
    F_, CA, MASS, LES = map(arr, (F_, CA, MASS, LES))
    return dict(F=F_, CA=CA, MASS=MASS, LES=LES, CFS=F_ * CA,
                probs=arr(probs), y=np.array(ys, int))


def _fmt(m, lo, hi):
    return f"{m:.3f} [{lo:.3f},{hi:.3f}]"


def summarize(name, R):
    fm, flo, fhi = M.bootstrap_ci(R["F"]); cm, clo, chi = M.bootstrap_ci(R["CA"])
    mm, mlo, mhi = M.bootstrap_ci(R["MASS"]); sm, slo, shi = M.bootstrap_ci(R["CFS"])
    auroc = roc_auc_score(R["y"], R["probs"]) if len(np.unique(R["y"])) > 1 else 0.5
    return {name: dict(
        malignancy_AUROC=round(float(auroc), 3),
        faithfulness=_fmt(fm, flo, fhi),
        concept_alignment_AUCIoU=_fmt(cm, clo, chi),
        concept_mass_in_band=_fmt(mm, mlo, mhi),
        CFS=_fmt(sm, slo, shi),
        lesion_overlap_descriptor=round(float(R["LES"].mean()), 3))}


def paired_block(Rc, Rv):
    """Paired CFA - baseline differences with bootstrap CI + Wilcoxon p."""
    out = {}
    for key, label in [("F", "faithfulness"), ("CA", "concept_alignment_AUCIoU"),
                       ("MASS", "concept_mass_in_band"), ("CFS", "CFS")]:
        d, lo, hi, p = M.paired_bootstrap_delta(Rc[key], Rv[key])
        out[label] = dict(delta_CFA_minus_base=round(d, 4),
                          ci95=[round(lo, 4), round(hi, 4)],
                          bootstrap_p=round(p, 4),
                          wilcoxon_p=round(M.wilcoxon_p(Rc[key], Rv[key]), 4),
                          significant=bool(lo > 0 or hi < 0))
    a1, a2, da, lo, hi, p = M.auroc_paired_bootstrap(Rc["y"], Rc["probs"], Rv["probs"])
    out["malignancy_AUROC"] = dict(delta_CFA_minus_base=round(da, 4),
                                   ci95=[round(lo, 4), round(hi, 4)],
                                   bootstrap_p=round(p, 4),
                                   significant=bool(lo > 0 or hi < 0))
    return out


def forest_plot(paired, n_test, path):
    labels = ["faithfulness", "concept_mass_in_band", "concept_alignment_AUCIoU",
              "CFS", "malignancy_AUROC"]
    labels = [k for k in labels if k in paired]
    deltas = [paired[k]["delta_CFA_minus_base"] for k in labels]
    los = [paired[k]["ci95"][0] for k in labels]
    his = [paired[k]["ci95"][1] for k in labels]
    yy = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, k in zip(yy, labels):
        sig = paired[k]["significant"]
        ax.plot([paired[k]["ci95"][0], paired[k]["ci95"][1]], [i, i],
                color=("tab:green" if sig else "tab:gray"), lw=2.5, zorder=2)
    ax.scatter(deltas, yy, color="black", zorder=3, s=30)
    ax.axvline(0, color="crimson", ls="--", lw=1)
    ax.set_yticks(yy); ax.set_yticklabels(labels)
    ax.set_xlabel("paired difference  (CFA − baseline),  95% bootstrap CI")
    ax.set_title(f"CF-Nodule: paired CFA effect on real LIDC test set (n={n_test})\n"
                 "green = CI excludes 0 (significant); dashed = no effect")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True, help="CFA model checkpoint")
    ap.add_argument("--baseline", help="non-CFA checkpoint (ablation)")
    ap.add_argument("--outdir", default="runs/eval")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(a.config))
    _, _, te = make_datasets(cfg)
    steps, n_th = cfg["eval"]["faithfulness_steps"], cfg["eval"]["auc_iou_thresholds"]

    m_cfa, _ = load(a.ckpt, dev)
    Rc = assess(m_cfa, te, dev, steps, n_th)
    summary = {"n_test": int(len(Rc["y"])),
               "test_class_balance": {"pos": int(Rc["y"].sum()),
                                      "neg": int((Rc["y"] == 0).sum())}}
    summary.update(summarize("CFA (ours)", Rc))
    Rv = None
    if a.baseline:
        m_van, _ = load(a.baseline, dev)
        Rv = assess(m_van, te, dev, steps, n_th)
        summary.update(summarize("Vanilla", Rv))
        summary["paired_CFA_minus_baseline"] = paired_block(Rc, Rv)

    json.dump(summary, open(os.path.join(a.outdir, "metrics.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))

    # risk-coverage
    thr_c = M.best_threshold(Rc["y"], Rc["probs"])
    cov_c, acc_c = M.risk_coverage(Rc["y"], Rc["probs"], np.abs(Rc["probs"] - thr_c), thr_c)
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(cov_c, acc_c, label="CFA model (ours)")
    if Rv is not None:
        thr_v = M.best_threshold(Rv["y"], Rv["probs"])
        cov_v, acc_v = M.risk_coverage(Rv["y"], Rv["probs"], np.abs(Rv["probs"] - thr_v), thr_v)
        plt.plot(cov_v, acc_v, label="Vanilla model")
    plt.xlabel("coverage"); plt.ylabel("accuracy on covered cases")
    plt.title("Selective prediction (confidence abstention)"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "risk_coverage.png"), dpi=130); plt.close()

    if Rv is not None:
        forest_plot(summary["paired_CFA_minus_baseline"], len(Rc["y"]),
                    os.path.join(a.outdir, "paired_effects.png"))
    print("[ok] ->", a.outdir)


if __name__ == "__main__":
    main()
