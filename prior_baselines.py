"""Prior-paradigm baselines under OUR identical 5-fold protocol, for fair AUROCs:
   - cnn_classifier : image -> malignancy only (multi-crop / plain CNN paradigm)
   - hscnn_style    : malignancy from [features ; predicted concepts] (HSCNN-style)
"""
import json, copy, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from src.models.cf_nodule import CFNodule
from src.losses import total_loss
from kfold import load_npz, fold_datasets, K

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = yaml.safe_load(open("configs/default.yaml"))
npz = load_npz(cfg)


def run(mal_mode, w_concept, w_seg, seed_base=3000):
    aurocs = []
    for fold in range(K):
        tr, va, te = fold_datasets(npz, cfg, fold)
        c = copy.deepcopy(cfg)
        c["train"]["w_cfa"] = 0.0; c["train"]["w_concept"] = w_concept; c["train"]["w_seg"] = w_seg
        torch.manual_seed(seed_base + fold); np.random.seed(seed_base + fold)
        tl = DataLoader(tr, batch_size=c["train"]["batch_size"], shuffle=True, drop_last=True)
        vl = DataLoader(va, batch_size=c["train"]["batch_size"])
        model = CFNodule(n_concepts=7, base=c["model"]["base_channels"],
                         depth=c["model"]["depth"], mal_mode=mal_mode).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=c["train"]["lr"],
                                weight_decay=c["train"]["weight_decay"])
        best, bs, pat = 0.0, None, 0
        for ep in range(c["train"]["epochs"]):
            model.train()
            for batch in tl:
                batch = [b.to(dev) for b in batch]; img = batch[0]
                opt.zero_grad()
                loss, _ = total_loss(model(img), batch, model, img, c, ep)
                loss.backward(); opt.step()
            model.eval(); ys, ps = [], []
            with torch.no_grad():
                for img, m, cc, r, y in vl:
                    ps.append(torch.sigmoid(model(img.to(dev))["malignancy"]).cpu().numpy()); ys.append(y.numpy())
            ys, ps = np.concatenate(ys), np.concatenate(ps)
            a = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else 0.5
            if a > best: best, bs, pat = a, copy.deepcopy(model.state_dict()), 0
            else:
                pat += 1
                if pat >= c["train"]["early_stop_patience"]: break
        model.load_state_dict(bs)
        model.eval(); ys, ps = [], []
        for img, m, cc, r, y in DataLoader(te, batch_size=64):
            with torch.no_grad():
                ps.append(torch.sigmoid(model(img.to(dev))["malignancy"]).cpu().numpy())
            ys.append(y.numpy())
        ys, ps = np.concatenate(ys), np.concatenate(ps)
        aurocs.append(float(roc_auc_score(ys, ps)) if len(np.unique(ys)) > 1 else 0.5)
        print(f"  [{mal_mode} wc={w_concept} ws={w_seg}] fold {fold}: AUROC={aurocs[-1]:.3f}", flush=True)
    return np.array(aurocs)


res = {}
res["cnn_classifier"] = run("features", 0.0, 0.0)
res["hscnn_style"] = run("augmented", 0.5, 1.0)
summary = {k: {"mean": round(float(v.mean()), 3), "std": round(float(v.std(ddof=1)), 3),
               "folds": [round(float(x), 3) for x in v]} for k, v in res.items()}
json.dump(summary, open("runs/prior_baselines.json", "w"), indent=2)
print("\n===== PRIOR-PARADIGM BASELINES (5-fold AUROC) =====")
print(json.dumps(summary, indent=2))
