"""
Multi-seed validation of the CFA effect on real LIDC -- the rigor that turns a
single noisy run into a publishable claim.

For each seed we make a fresh patient-level split, train CFA-on and CFA-off under
identical conditions, evaluate both on the SAME test split (paired), and record the
per-metric mean difference (CFA - baseline). We then test whether the across-seed
mean difference is stable / non-zero (one-sample t-test + sign count).

    python multiseed.py            # uses configs/default.yaml, 5 seeds
"""
import sys, json, copy, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.models.cf_nodule import CFNodule
from src.losses import total_loss
from src.data.lidc_dataset import make_datasets
from src.evaluate import assess

SEEDS = [7, 13, 21, 42, 101]
METRICS = [("F", "faithfulness"), ("CA", "concept_alignment"),
           ("MASS", "concept_mass_in_band"), ("CFS", "CFS")]


def train_one(cfg, dev, cfa_on):
    cfg = copy.deepcopy(cfg)
    if not cfa_on:
        cfg["train"]["w_cfa"] = 0.0
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    tr, va, te = make_datasets(cfg)
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
            out = model(img)
            loss, _ = total_loss(out, batch, model, img, cfg, epoch)
            loss.backward(); opt.step()
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for img, mask, c, r, y in vl:
                ps.append(torch.sigmoid(model(img.to(dev))["malignancy"]).cpu().numpy())
                ys.append(y.numpy())
        ys, ps = np.concatenate(ys), np.concatenate(ps)
        auroc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else 0.5
        if auroc > best:
            best, best_state, patience = auroc, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                break
    model.load_state_dict(best_state)
    return model, te


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg0 = yaml.safe_load(open("configs/default.yaml"))
    steps = cfg0["eval"]["faithfulness_steps"]; n_th = cfg0["eval"]["auc_iou_thresholds"]
    agg = {k: [] for k, _ in METRICS}
    auroc_c, auroc_v, per_seed = [], [], []
    for s in SEEDS:
        cfg = copy.deepcopy(cfg0); cfg["seed"] = s
        mc, te = train_one(cfg, dev, True)
        mv, _ = train_one(cfg, dev, False)
        Rc = assess(mc, te, dev, steps, n_th)
        Rv = assess(mv, te, dev, steps, n_th)
        row = {"seed": s, "n_test": int(len(Rc["y"]))}
        for k, name in METRICS:
            d = float((Rc[k] - Rv[k]).mean()); agg[k].append(d); row[name] = round(d, 4)
        ac = roc_auc_score(Rc["y"], Rc["probs"]); av = roc_auc_score(Rv["y"], Rv["probs"])
        auroc_c.append(ac); auroc_v.append(av)
        row["auroc_cfa"] = round(ac, 3); row["auroc_van"] = round(av, 3)
        per_seed.append(row)
        print("seed", s, row, flush=True)

    from scipy.stats import ttest_1samp
    summary = {}
    for k, name in METRICS:
        a = np.array(agg[k])
        t = ttest_1samp(a, 0.0)
        summary[name] = dict(mean_delta=round(float(a.mean()), 4),
                             std=round(float(a.std(ddof=1)), 4),
                             seeds_positive=f"{int((a > 0).sum())}/{len(a)}",
                             t_p=round(float(t.pvalue), 4),
                             stable_significant=bool(t.pvalue < 0.05 and (a > 0).all() or (a < 0).all()))
    summary["malignancy_AUROC"] = dict(
        cfa_mean=round(float(np.mean(auroc_c)), 3), van_mean=round(float(np.mean(auroc_v)), 3),
        delta_mean=round(float(np.mean(np.array(auroc_c) - np.array(auroc_v))), 4))
    out = {"seeds": SEEDS, "per_seed": per_seed, "across_seed_delta": summary}
    json.dump(out, open("runs/multiseed_summary.json", "w"), indent=2)
    print("\n===== ACROSS-SEED SUMMARY (CFA - baseline) =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
