"""
Comprehensive benchmark for CF-Nodule on real LIDC (700 patients), 5-fold patient-CV.

Comparison methods + ablations, all under one code path and identical folds/inits, so
the numbers go straight into the paper's tables. Writes runs/benchmark.json
INCREMENTALLY and skips runs already present (resumable).

    python benchmark.py
"""
import json, os, copy, yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from scipy.stats import ttest_1samp, ttest_rel

from src.models.cf_nodule import CFNodule
from src.losses import dice_loss
from kfold import load_npz, fold_datasets, K

CONCEPTS = ["subtlety", "calcification", "sphericity", "margin",
            "lobulation", "spiculation", "texture"]
CLINICAL3 = [3, 4, 5]                         # margin, lobulation, spiculation
ALL7 = list(range(7))
OUT = "runs/benchmark.json"

# (name, type, kwargs) -- type drives training; kwargs tune model/loss/concepts
RUNS = [
    # ---- comparison methods ----
    ("concept_only_trueMLP", "concept_only", dict(cidx=ALL7)),
    ("blackbox",             "blackbox",     dict(cidx=ALL7)),
    ("cbm_joint",            "joint",        dict(cidx=ALL7, hidden=32)),
    ("cbm_sequential",       "sequential",   dict(cidx=ALL7, hidden=32)),
    ("cbm_independent_MAIN", "independent",  dict(cidx=ALL7, hidden=32, w_concept=0.5)),
    # ---- ablations on the independent CBM (vary one axis from MAIN) ----
    ("indep_head_linear",    "independent",  dict(cidx=ALL7, hidden=0,  w_concept=0.5)),
    ("indep_head_mlp64",     "independent",  dict(cidx=ALL7, hidden=64, w_concept=0.5)),
    ("indep_concepts_clin3", "independent",  dict(cidx=CLINICAL3, hidden=32, w_concept=0.5)),
    ("indep_wconcept_0.25",  "independent",  dict(cidx=ALL7, hidden=32, w_concept=0.25)),
    ("indep_wconcept_1.0",   "independent",  dict(cidx=ALL7, hidden=32, w_concept=1.0)),
]


def build(cfg, kw):
    return CFNodule(n_concepts=len(kw["cidx"]), base=cfg["model"]["base_channels"],
                    depth=cfg["model"]["depth"],
                    concept_bottleneck=(kw.get("_bottleneck", True)),
                    mal_concept_hidden=kw.get("hidden", 32))


def _val_auroc(model, vl, dev):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for img, mask, c, r, y in vl:
            ps.append(torch.sigmoid(model(img.to(dev))["malignancy"]).cpu().numpy()); ys.append(y.numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    return roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else 0.5


def _loaders(cfg, tr, va):
    return (DataLoader(tr, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True),
            DataLoader(va, batch_size=cfg["train"]["batch_size"]))


def train_generic(cfg, dev, tr, va, kw, mode, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cidx = kw["cidx"]; wc = kw.get("w_concept", cfg["train"]["w_concept"])
    ws, wm = cfg["train"]["w_seg"], cfg["train"]["w_malignancy"]
    bottleneck = mode in ("joint", "sequential", "independent")
    model = CFNodule(n_concepts=len(cidx), base=cfg["model"]["base_channels"],
                     depth=cfg["model"]["depth"], concept_bottleneck=bottleneck,
                     mal_concept_hidden=kw.get("hidden", 32)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    tl, vl = _loaders(cfg, tr, va)
    best, best_state, patience = 0.0, None, 0
    phase2_start = cfg["train"]["epochs"] // 2 if mode == "sequential" else -1
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for img, mask, concept, region, y in tl:
            img, mask, y = img.to(dev), mask.to(dev), y.to(dev)
            concept = concept.to(dev)[:, cidx]
            opt.zero_grad()
            out = model(img)
            L = ws * dice_loss(out["seg"], mask) + wc * F.mse_loss(out["concepts"], concept)
            if mode == "blackbox" or mode == "joint":
                L = L + wm * F.binary_cross_entropy_with_logits(out["malignancy"], y.float())
            elif mode == "independent":
                L = L + wm * F.binary_cross_entropy_with_logits(
                    model.malignancy_from_concepts(concept), y.float())
            elif mode == "sequential":
                if epoch >= phase2_start:       # phase 2: mal head on PREDICTED concepts (detached)
                    L = L + wm * F.binary_cross_entropy_with_logits(
                        model.malignancy_from_concepts(out["concepts"].detach()), y.float())
            L.backward(); opt.step()
        a = _val_auroc(model, vl, dev)
        if a > best:
            best, best_state, patience = a, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"] and epoch > phase2_start:
                break
    model.load_state_dict(best_state); return model


def eval_model(model, te, dev, cidx, is_cbm):
    loader = DataLoader(te, batch_size=64); ys, pp, pt, cmae = [], [], [], []
    grads = np.zeros(len(cidx)); model.eval()
    for img, mask, concept, region, y in loader:
        img = img.to(dev); concept = concept.to(dev)[:, cidx]
        with torch.no_grad():
            out = model(img)
            pp.append(torch.sigmoid(out["malignancy"]).cpu().numpy())
            cmae.append(torch.abs(out["concepts"] - concept).mean().item())
            if is_cbm:
                pt.append(torch.sigmoid(model.malignancy_from_concepts(concept)).cpu().numpy())
        ys.append(y.numpy())
        if is_cbm:
            c = concept.clone().requires_grad_(True)
            model.malignancy_from_concepts(c).sum().backward()
            grads += np.abs(c.grad.cpu().numpy()).sum(0)
    ys = np.concatenate(ys); pp = np.concatenate(pp)
    r = dict(auroc=float(roc_auc_score(ys, pp)) if len(np.unique(ys)) > 1 else 0.5,
             concept_mae=float(np.mean(cmae)))
    if is_cbm:
        pt = np.concatenate(pt)
        r["auroc_intervention"] = float(roc_auc_score(ys, pt)) if len(np.unique(ys)) > 1 else 0.5
        r["importance"] = (grads / grads.sum()).tolist()
    return r


def run_concept_only(cfg, npz, kw):
    cidx = kw["cidx"]; aurocs = []
    for fold in range(K):
        tr, va, te = fold_datasets(npz, cfg, fold)
        def XY(ds):
            C = torch.stack([ds[i][2] for i in range(len(ds))]).numpy()[:, cidx]
            Y = np.array([int(ds[i][4]) for i in range(len(ds))])
            return C, Y
        Xtr, Ytr = XY(tr); Xte, Yte = XY(te)
        clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=0).fit(Xtr, Ytr)
        p = clf.predict_proba(Xte)[:, 1]
        aurocs.append(float(roc_auc_score(Yte, p)) if len(np.unique(Yte)) > 1 else 0.5)
    return [dict(fold=f, auroc=round(a, 3)) for f, a in enumerate(aurocs)]


def run_cv(cfg, dev, npz, name, mode, kw):
    cidx = kw["cidx"]; is_cbm = mode in ("joint", "sequential", "independent")
    rows, imp = [], np.zeros(len(cidx))
    for fold in range(K):
        tr, va, te = fold_datasets(npz, cfg, fold)
        model = train_generic(cfg, dev, tr, va, kw, mode, seed=2000 + fold)
        R = eval_model(model, te, dev, cidx, is_cbm)
        row = dict(fold=fold, auroc=round(R["auroc"], 3), concept_mae=round(R["concept_mae"], 3))
        if is_cbm:
            row["auroc_intervention"] = round(R["auroc_intervention"], 3)
            imp += np.array(R["importance"])
        rows.append(row); print(f"  [{name}] fold {fold}: {row}", flush=True)
    out = {"per_fold": rows}
    if is_cbm:
        out["concept_importance"] = {CONCEPTS[c]: round(float(w), 3)
                                     for c, w in zip(cidx, imp / imp.sum())}
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open("configs/default.yaml"))
    npz = load_npz(cfg)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name, mode, kw in RUNS:
        if name in results:
            print(f"[skip] {name} (already done)"); continue
        print(f"[run] {name} ({mode}) {kw}", flush=True)
        if mode == "concept_only":
            res = {"per_fold": run_concept_only(cfg, npz, kw), "mode": mode}
        else:
            res = run_cv(cfg, dev, npz, name, mode, kw); res["mode"] = mode
        res["kwargs"] = {k: v for k, v in kw.items() if k != "cidx"}
        res["n_concepts"] = len(kw["cidx"])
        results[name] = res
        json.dump(results, open(OUT, "w"), indent=2)     # incremental save

    # ---- build comparison table with paired tests vs blackbox ----
    def folds(name, key="auroc"):
        return np.array([r[key] for r in results[name]["per_fold"]])
    base = folds("blackbox")
    table = {}
    for name in results:
        a = folds(name)
        entry = dict(auroc_mean=round(float(a.mean()), 3), auroc_std=round(float(a.std(ddof=1)), 3))
        if name != "blackbox" and len(a) == len(base):
            entry["delta_vs_blackbox"] = round(float((a - base).mean()), 4)
            entry["paired_t_p_vs_blackbox"] = round(float(ttest_rel(a, base).pvalue), 4)
        if "auroc_intervention" in results[name]["per_fold"][0]:
            iv = folds(name, "auroc_intervention")
            entry["auroc_intervention_mean"] = round(float(iv.mean()), 3)
        if "concept_importance" in results[name]:
            entry["concept_importance"] = results[name]["concept_importance"]
        table[name] = entry
    results["_comparison_table"] = table
    json.dump(results, open(OUT, "w"), indent=2)
    print("\n===== COMPARISON / ABLATION TABLE =====")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
