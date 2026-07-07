"""Improve the proposed model's malignancy AUROC, honestly, under the same 5-fold CV:
   - indep_improved : independent CBM + flip/rot augmentation + stronger head (faithful)
   - hybrid         : concepts + residual feature channel (higher acc, partial faithfulness)
Reports AUROC (+ true-concept intervention for the faithful variant).
"""
import json, copy, yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from src.models.cf_nodule import CFNodule
from src.losses import dice_loss
from kfold import load_npz, fold_datasets, K

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = yaml.safe_load(open("configs/default.yaml"))
npz = load_npz(cfg)


def augment(img, mask, region):
    if torch.rand(1).item() < 0.5: img, mask, region = [torch.flip(t, [-1]) for t in (img, mask, region)]
    if torch.rand(1).item() < 0.5: img, mask, region = [torch.flip(t, [-2]) for t in (img, mask, region)]
    k = int(torch.randint(0, 4, (1,)).item())
    if k: img, mask, region = [torch.rot90(t, k, [-2, -1]) for t in (img, mask, region)]
    return img, mask, region


def run(mode, w_concept, hidden, aug, epochs, patience, seed_base=4000):
    aurocs, intervs = [], []
    for fold in range(K):
        tr, va, te = fold_datasets(npz, cfg, fold)
        torch.manual_seed(seed_base + fold); np.random.seed(seed_base + fold)
        tl = DataLoader(tr, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)
        vl = DataLoader(va, batch_size=cfg["train"]["batch_size"])
        model = CFNodule(n_concepts=7, base=cfg["model"]["base_channels"], depth=cfg["model"]["depth"],
                         mal_mode=mode, mal_concept_hidden=hidden).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
        best, bs, pat = 0.0, None, 0
        for ep in range(epochs):
            model.train()
            for img, mask, concept, region, y in tl:
                img, mask, region = img.to(dev), mask.to(dev), region.to(dev)
                concept, y = concept.to(dev), y.to(dev)
                if aug: img, mask, region = augment(img, mask, region)
                opt.zero_grad()
                out = model(img)
                L = w_concept * F.mse_loss(out["concepts"], concept) + dice_loss(out["seg"], mask)
                if mode == "concepts":                      # independent: train head on TRUE concepts
                    L = L + F.binary_cross_entropy_with_logits(model.malignancy_from_concepts(concept), y.float())
                else:                                        # augmented / features: forward malignancy
                    L = L + F.binary_cross_entropy_with_logits(out["malignancy"], y.float())
                L.backward(); opt.step()
            model.eval(); ys, ps = [], []
            with torch.no_grad():
                for img, m, cc, r, y in vl:
                    ps.append(torch.sigmoid(model(img.to(dev))["malignancy"]).cpu().numpy()); ys.append(y.numpy())
            ys, ps = np.concatenate(ys), np.concatenate(ps)
            a = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else 0.5
            if a > best: best, bs, pat = a, copy.deepcopy(model.state_dict()), 0
            else:
                pat += 1
                if pat >= patience: break
        model.load_state_dict(bs)
        model.eval(); ys, ps, pt = [], [], []
        for img, m, cc, r, y in DataLoader(te, batch_size=64):
            img = img.to(dev); cc = cc.to(dev)
            with torch.no_grad():
                ps.append(torch.sigmoid(model(img)["malignancy"]).cpu().numpy())
                if mode == "concepts":
                    pt.append(torch.sigmoid(model.malignancy_from_concepts(cc)).cpu().numpy())
            ys.append(y.numpy())
        ys, ps = np.concatenate(ys), np.concatenate(ps)
        aurocs.append(float(roc_auc_score(ys, ps)) if len(np.unique(ys)) > 1 else 0.5)
        if mode == "concepts":
            pt = np.concatenate(pt); intervs.append(float(roc_auc_score(ys, pt)) if len(np.unique(ys)) > 1 else 0.5)
        print(f"  [{mode} aug={aug} h={hidden}] fold {fold}: AUROC={aurocs[-1]:.3f}", flush=True)
    out = {"auroc_mean": round(float(np.mean(aurocs)), 3), "auroc_std": round(float(np.std(aurocs, ddof=1)), 3),
           "folds": [round(x, 3) for x in aurocs]}
    if intervs: out["intervention_mean"] = round(float(np.mean(intervs)), 3)
    return out


res = {}
res["indep_improved"] = run("concepts", w_concept=1.0, hidden=128, aug=True, epochs=120, patience=25)
res["hybrid_residual"] = run("augmented", w_concept=1.0, hidden=32, aug=True, epochs=120, patience=25)
json.dump(res, open("runs/improve_auroc.json", "w"), indent=2)
print("\n===== IMPROVED AUROC (5-fold) =====")
print(json.dumps(res, indent=2))
