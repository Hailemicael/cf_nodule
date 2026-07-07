"""
INDEPENDENT concept-bottleneck CF-Nodule -- the leakage-free CBM, 5-fold CV.

Joint CBM leaks: its malignancy head reads predicted-concept activations (which
smuggle feature info), so the concept-intervention test fails (true concepts ->
chance AUROC). Fix: train the malignancy head on GROUND-TRUTH concepts, so it must
learn genuine concept->malignancy semantics. At test it consumes PREDICTED concepts.
This is faithful by construction; intervention with true concepts gives the upper
bound, and test accuracy is bottlenecked by concept-prediction quality (the honest
faithfulness/accuracy trade-off).

Baseline AUROC for comparison is read from runs/cbm_summary.json (same folds/inits).

    python cbm_independent.py
"""
import json, copy, yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.models.cf_nodule import CFNodule
from src.losses import dice_loss
from src import metrics as M
from kfold import load_npz, fold_datasets, K

CONCEPTS = ["subtlety", "calcification", "sphericity", "margin",
            "lobulation", "spiculation", "texture"]


def train_independent(cfg, dev, tr, va, init_seed):
    torch.manual_seed(init_seed); np.random.seed(init_seed)
    tl = DataLoader(tr, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=cfg["train"]["batch_size"])
    model = CFNodule(n_concepts=len(cfg["data"]["concepts"]), base=cfg["model"]["base_channels"],
                     depth=cfg["model"]["depth"], concept_bottleneck=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    ws, wc, wm = cfg["train"]["w_seg"], cfg["train"]["w_concept"], cfg["train"]["w_malignancy"]
    best, best_state, patience = 0.0, None, 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for img, mask, concept, region, y in tl:
            img, mask, concept, y = img.to(dev), mask.to(dev), concept.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(img)
            L_seg = dice_loss(out["seg"], mask)
            L_con = F.mse_loss(out["concepts"], concept)
            # INDEPENDENT: malignancy head trained on TRUE concepts (no leakage)
            mal_true = model.malignancy_from_concepts(concept)
            L_mal = F.binary_cross_entropy_with_logits(mal_true, y.float())
            (ws * L_seg + wc * L_con + wm * L_mal).backward()
            opt.step()
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for img, mask, c, r, y in vl:
                ps.append(torch.sigmoid(model(img.to(dev))["malignancy"]).cpu().numpy()); ys.append(y.numpy())
        ys, ps = np.concatenate(ys), np.concatenate(ps)
        auroc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else 0.5
        if auroc > best:
            best, best_state, patience = auroc, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                break
    model.load_state_dict(best_state); return model


def evaluate(model, te, dev):
    loader = DataLoader(te, batch_size=64); ys, pp, pt, cmae = [], [], [], []
    grads = np.zeros(len(CONCEPTS)); model.eval()
    for img, mask, concept, region, y in loader:
        img, concept = img.to(dev), concept.to(dev)
        with torch.no_grad():
            out = model(img)
            pp.append(torch.sigmoid(out["malignancy"]).cpu().numpy())
            pt.append(torch.sigmoid(model.malignancy_from_concepts(concept)).cpu().numpy())
            cmae.append(torch.abs(out["concepts"] - concept).mean().item())
        ys.append(y.numpy())
        c = concept.clone().requires_grad_(True)
        model.malignancy_from_concepts(c).sum().backward()
        grads += np.abs(c.grad.cpu().numpy()).sum(0)
    ys = np.concatenate(ys)
    return dict(y=ys, p_pred=np.concatenate(pp), p_true=np.concatenate(pt),
                auroc=roc_auc_score(ys, np.concatenate(pp)) if len(np.unique(ys)) > 1 else 0.5,
                auroc_intervention=roc_auc_score(ys, np.concatenate(pt)) if len(np.unique(ys)) > 1 else 0.5,
                concept_mae=float(np.mean(cmae)), importance=(grads / grads.sum()).tolist())


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open("configs/default.yaml"))
    npz = load_npz(cfg)
    base = {r["fold"]: r["auroc_baseline"] for r in json.load(open("runs/cbm_summary.json"))["per_fold"]}
    per_fold, imp = [], np.zeros(len(CONCEPTS))
    for fold in range(K):
        tr, va, te = fold_datasets(npz, cfg, fold)
        m = train_independent(cfg, dev, tr, va, 2000 + fold)
        R = evaluate(m, te, dev)
        row = dict(fold=fold, n_test=int(len(R["y"])),
                   auroc_indep_cbm=round(R["auroc"], 3), auroc_baseline=base.get(fold),
                   auroc_intervention_truec=round(R["auroc_intervention"], 3),
                   concept_mae=round(R["concept_mae"], 3))
        per_fold.append(row); imp += np.array(R["importance"]); print("fold", fold, row, flush=True)

    from scipy.stats import ttest_1samp
    d_base = np.array([r["auroc_indep_cbm"] - r["auroc_baseline"] for r in per_fold])
    d_intv = np.array([r["auroc_intervention_truec"] - r["auroc_indep_cbm"] for r in per_fold])
    imp = imp / imp.sum()
    summary = dict(
        auroc_indep_mean=round(float(np.mean([r["auroc_indep_cbm"] for r in per_fold])), 3),
        auroc_baseline_mean=round(float(np.mean([r["auroc_baseline"] for r in per_fold])), 3),
        delta_indep_minus_base=dict(mean=round(float(d_base.mean()), 4),
                                    std=round(float(d_base.std(ddof=1)), 4),
                                    t_p=round(float(ttest_1samp(d_base, 0).pvalue), 4)),
        intervention_gain_truec=dict(mean=round(float(d_intv.mean()), 4),
                                     std=round(float(d_intv.std(ddof=1)), 4),
                                     folds_positive=f"{int((d_intv > 0).sum())}/{K}",
                                     t_p=round(float(ttest_1samp(d_intv, 0).pvalue), 4)),
        concept_importance={c: round(float(w), 3) for c, w in zip(CONCEPTS, imp)})
    out = dict(protocol=f"{K}-fold CV, INDEPENDENT CBM", per_fold=per_fold, summary=summary)
    json.dump(out, open("runs/cbm_independent_summary.json", "w"), indent=2)
    print("\n===== INDEPENDENT CBM SUMMARY =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
