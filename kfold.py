"""
5-fold cross-validated, PAIRED evaluation of the CFA effect on real LIDC.

This is the well-powered, honest protocol:
  - patient-level 5-fold split (every nodule is in the test fold exactly once);
  - per fold, train CFA-on and CFA-off from the SAME init on the same train data;
  - score both on the held-out fold (paired, per nodule);
  - POOL the per-nodule paired differences across folds (n ~ all nodules) and run a
    paired bootstrap CI + Wilcoxon test -- far more power than 5 seed-means;
  - ALSO report the conservative per-fold mean deltas.

    python kfold.py
"""
import json, copy, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from src.models.cf_nodule import CFNodule
from src.losses import total_loss
from src import metrics as M
from src.evaluate import assess

K = 5
FOLD_SEED = 7
METRICS = [("F", "faithfulness"), ("CA", "concept_alignment"),
           ("MASS", "concept_mass_in_band"), ("CFS", "CFS")]


def load_npz(cfg):
    return np.load(f"{cfg['data']['cache']}/lidc_patches.npz", allow_pickle=True)


def fold_datasets(npz, cfg, fold):
    pid = npz["pid"]
    upid = np.unique(pid)
    np.random.default_rng(FOLD_SEED).shuffle(upid)
    folds = np.array_split(upid, K)
    test_p = set(folds[fold].tolist())
    rest = [p for p in upid if p not in test_p]
    n_val = max(1, int(0.15 * len(rest)))
    val_p, train_p = set(rest[:n_val]), set(rest[n_val:])

    def idx(group):
        return np.array([i for i, p in enumerate(pid) if p in group])

    def ds(ix):
        t = lambda k: torch.tensor(npz[k][ix])
        return TensorDataset(t("image").float(), t("mask").float(),
                             t("concept").float(), t("region").float(),
                             torch.tensor(npz["y"][ix]).long())
    return ds(idx(train_p)), ds(idx(val_p)), ds(idx(test_p))


def train_model(cfg, dev, tr, va, cfa_on, init_seed):
    cfg = copy.deepcopy(cfg)
    if not cfa_on:
        cfg["train"]["w_cfa"] = 0.0
    torch.manual_seed(init_seed); np.random.seed(init_seed)   # identical init for fair pair
    tl = DataLoader(tr, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=cfg["train"]["batch_size"])
    model = CFNodule(n_concepts=len(cfg["data"]["concepts"]),
                     base=cfg["model"]["base_channels"], depth=cfg["model"]["depth"]).to(dev)
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
    model.load_state_dict(best_state)
    return model


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open("configs/default.yaml"))
    steps, n_th = cfg["eval"]["faithfulness_steps"], cfg["eval"]["auc_iou_thresholds"]
    npz = load_npz(cfg)

    pooled = {k: [] for k, _ in METRICS}
    pooled_y, pooled_pc, pooled_pv = [], [], []
    per_fold = []
    for fold in range(K):
        tr, va, te = fold_datasets(npz, cfg, fold)
        mc = train_model(cfg, dev, tr, va, True, init_seed=1000 + fold)
        mv = train_model(cfg, dev, tr, va, False, init_seed=1000 + fold)
        Rc = assess(mc, te, dev, steps, n_th)
        Rv = assess(mv, te, dev, steps, n_th)
        row = {"fold": fold, "n_test": int(len(Rc["y"]))}
        for k, name in METRICS:
            d = Rc[k] - Rv[k]
            pooled[k].extend(d.tolist())
            row[name] = round(float(d.mean()), 4)
        pooled_y.extend(Rc["y"].tolist())
        pooled_pc.extend(Rc["probs"].tolist()); pooled_pv.extend(Rv["probs"].tolist())
        row["auroc_cfa"] = round(roc_auc_score(Rc["y"], Rc["probs"]), 3)
        row["auroc_van"] = round(roc_auc_score(Rv["y"], Rv["probs"]), 3)
        per_fold.append(row); print("fold", fold, row, flush=True)

    # pooled per-nodule paired analysis (well-powered)
    pooled_stats = {}
    for k, name in METRICS:
        a = np.array(pooled[k])
        delta, lo, hi, p = M.paired_bootstrap_delta(a, np.zeros_like(a))
        pooled_stats[name] = dict(n=len(a), mean_delta=round(float(a.mean()), 4),
                                  ci95=[round(lo, 4), round(hi, 4)],
                                  bootstrap_p=round(p, 4),
                                  wilcoxon_p=round(M.wilcoxon_p(a, np.zeros_like(a)), 4),
                                  significant=bool(lo > 0 or hi < 0))
    yy = np.array(pooled_y); pc = np.array(pooled_pc); pv = np.array(pooled_pv)
    a1, a2, da, lo, hi, p = M.auroc_paired_bootstrap(yy, pc, pv)
    pooled_stats["malignancy_AUROC"] = dict(cfa=round(a1, 3), van=round(a2, 3),
                                            delta=round(da, 4), ci95=[round(lo, 4), round(hi, 4)],
                                            bootstrap_p=round(p, 4))
    # conservative per-fold mean test
    from scipy.stats import ttest_1samp
    fold_stats = {}
    for k, name in METRICS:
        vals = np.array([r[name] for r in per_fold])
        fold_stats[name] = dict(mean=round(float(vals.mean()), 4),
                                std=round(float(vals.std(ddof=1)), 4),
                                folds_positive=f"{int((vals > 0).sum())}/{K}",
                                t_p=round(float(ttest_1samp(vals, 0.0).pvalue), 4))

    out = {"protocol": f"{K}-fold patient-level CV, paired per-nodule",
           "per_fold": per_fold,
           "pooled_per_nodule": pooled_stats,
           "per_fold_mean_test": fold_stats}
    json.dump(out, open("runs/kfold_summary.json", "w"), indent=2)
    print("\n===== POOLED PER-NODULE (CFA - baseline) =====")
    print(json.dumps(pooled_stats, indent=2))
    print("\n===== PER-FOLD MEAN TEST (conservative) =====")
    print(json.dumps(fold_stats, indent=2))


if __name__ == "__main__":
    main()
