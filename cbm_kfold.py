"""
Concept-Bottleneck CF-Nodule vs black-box baseline -- 5-fold CV on real LIDC.

CBM: features -> predicted concepts -> malignancy (decision flows THROUGH concepts,
so it is concept-grounded + faithful BY CONSTRUCTION). Baseline: malignancy straight
from image features (concepts only an auxiliary head).

Per fold we train both from the SAME init on the same data and measure:
  - AUROC(CBM) vs AUROC(baseline): does the bottleneck cost accuracy?
  - CONCEPT INTERVENTION (CBM only): feed GROUND-TRUTH concepts -> malignancy AUROC.
    If true-concept AUROC > predicted-concept AUROC, the decision is genuinely
    concept-mediated (concept errors are the bottleneck) -- a faithfulness property
    a black box cannot have.
  - concept prediction MAE, and per-concept importance |d mal / d concept|.

    python cbm_kfold.py
"""
import json, copy, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.models.cf_nodule import CFNodule
from src.losses import total_loss
from src import metrics as M
from kfold import load_npz, fold_datasets, K

CONCEPTS = ["subtlety", "calcification", "sphericity", "margin",
            "lobulation", "spiculation", "texture"]


def train(cfg, dev, tr, va, bottleneck, init_seed):
    cfg = copy.deepcopy(cfg); cfg["train"]["w_cfa"] = 0.0     # no Grad-CAM CFA term
    torch.manual_seed(init_seed); np.random.seed(init_seed)
    tl = DataLoader(tr, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=cfg["train"]["batch_size"])
    model = CFNodule(n_concepts=len(cfg["data"]["concepts"]), base=cfg["model"]["base_channels"],
                     depth=cfg["model"]["depth"], concept_bottleneck=bottleneck).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    best, best_state, patience = 0.0, None, 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for batch in tl:
            batch = [b.to(dev) for b in batch]; img = batch[0]
            opt.zero_grad()
            loss, _ = total_loss(model(img), batch, model, img, cfg, epoch)
            loss.backward(); opt.step()
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


def evaluate_cbm(model, te, dev, is_cbm):
    loader = DataLoader(te, batch_size=64)
    ys, p_pred, p_true, cmae = [], [], [], []
    grads = np.zeros(len(CONCEPTS))
    model.eval()
    for img, mask, concept, region, y in loader:
        img = img.to(dev); concept = concept.to(dev)
        with torch.no_grad():
            out = model(img)
            p_pred.append(torch.sigmoid(out["malignancy"]).cpu().numpy())
            cmae.append(torch.abs(out["concepts"] - concept).mean(0).cpu().numpy())
            if is_cbm:
                p_true.append(torch.sigmoid(model.malignancy_from_concepts(concept)).cpu().numpy())
        ys.append(y.numpy())
        if is_cbm:                              # per-concept importance |d mal / d concept|
            c = concept.clone().requires_grad_(True)
            model.malignancy_from_concepts(c).sum().backward()
            grads += np.abs(c.grad.cpu().numpy()).sum(0)
    ys = np.concatenate(ys); p_pred = np.concatenate(p_pred)
    res = dict(y=ys, p_pred=p_pred,
               auroc=roc_auc_score(ys, p_pred) if len(np.unique(ys)) > 1 else 0.5,
               concept_mae=float(np.mean(np.concatenate([m[None] for m in cmae]).mean(0))))
    if is_cbm:
        res["p_true"] = np.concatenate(p_true)
        res["auroc_intervention"] = roc_auc_score(ys, res["p_true"]) if len(np.unique(ys)) > 1 else 0.5
        res["concept_importance"] = (grads / grads.sum()).tolist()
    return res


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open("configs/default.yaml"))
    npz = load_npz(cfg)
    per_fold, imp_acc = [], np.zeros(len(CONCEPTS))
    pool = dict(y=[], cbm=[], van=[], interv=[])
    for fold in range(K):
        tr, va, te = fold_datasets(npz, cfg, fold)
        m_cbm = train(cfg, dev, tr, va, True, 2000 + fold)
        m_van = train(cfg, dev, tr, va, False, 2000 + fold)
        Rc = evaluate_cbm(m_cbm, te, dev, True)
        Rv = evaluate_cbm(m_van, te, dev, False)
        row = dict(fold=fold, n_test=int(len(Rc["y"])),
                   auroc_cbm=round(Rc["auroc"], 3), auroc_baseline=round(Rv["auroc"], 3),
                   auroc_intervention_truec=round(Rc["auroc_intervention"], 3),
                   concept_mae_cbm=round(Rc["concept_mae"], 3))
        per_fold.append(row); print("fold", fold, row, flush=True)
        imp_acc += np.array(Rc["concept_importance"])
        pool["y"].extend(Rc["y"].tolist()); pool["cbm"].extend(Rc["p_pred"].tolist())
        pool["van"].extend(Rv["p_pred"].tolist()); pool["interv"].extend(Rc["p_true"].tolist())

    y = np.array(pool["y"]); pc = np.array(pool["cbm"]); pv = np.array(pool["van"]); pi = np.array(pool["interv"])
    a_cbm, a_van, d, lo, hi, p = M.auroc_paired_bootstrap(y, pc, pv)
    ai, ac2, di, loi, hii, pi_p = M.auroc_paired_bootstrap(y, pi, pc)   # intervention vs predicted
    from scipy.stats import ttest_1samp
    fold_d = np.array([r["auroc_cbm"] - r["auroc_baseline"] for r in per_fold])
    intv_d = np.array([r["auroc_intervention_truec"] - r["auroc_cbm"] for r in per_fold])
    imp = imp_acc / imp_acc.sum()
    summary = dict(
        pooled_AUROC=dict(cbm=round(a_cbm, 3), baseline=round(a_van, 3),
                          delta_cbm_minus_base=round(d, 4), ci95=[round(lo, 4), round(hi, 4)],
                          bootstrap_p=round(p, 4)),
        per_fold_AUROC_cbm_minus_base=dict(mean=round(float(fold_d.mean()), 4),
                                           std=round(float(fold_d.std(ddof=1)), 4),
                                           t_p=round(float(ttest_1samp(fold_d, 0).pvalue), 4)),
        concept_intervention=dict(
            auroc_true_concepts=round(ai, 3), auroc_pred_concepts=round(a_cbm, 3),
            delta=round(di, 4), ci95=[round(loi, 4), round(hii, 4)], bootstrap_p=round(pi_p, 4),
            per_fold_mean=round(float(intv_d.mean()), 4),
            per_fold_t_p=round(float(ttest_1samp(intv_d, 0).pvalue), 4)),
        concept_importance={c: round(float(w), 3) for c, w in zip(CONCEPTS, imp)})
    out = dict(protocol=f"{K}-fold CV, CBM vs baseline", per_fold=per_fold, summary=summary)
    json.dump(out, open("runs/cbm_summary.json", "w"), indent=2)
    print("\n===== CBM SUMMARY =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
