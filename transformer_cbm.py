"""Pretrained ViT encoder for the faithful (independent) CBM, 5-fold CV on real LIDC.
Backbone: timm vit_tiny_patch16_224 (ImageNet-pretrained). 64x64 grayscale patches
are resized to 224 and repeated to 3 channels. Concept head -> independent
malignancy head (trained on ground-truth concepts). Reports AUROC + intervention.
"""
import json, copy, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import timm
from kfold import load_npz, fold_datasets, K

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = yaml.safe_load(open("configs/default.yaml"))
npz = load_npz(cfg)
C = 7
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def prep(x):  # x: B,1,64,64 in [0,1] -> B,3,224,224 ImageNet-normalized
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    return (x - MEAN) / STD


def augment(x):
    if torch.rand(1).item() < 0.5: x = torch.flip(x, [-1])
    if torch.rand(1).item() < 0.5: x = torch.flip(x, [-2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k: x = torch.rot90(x, k, [-2, -1])
    return x


class ViTCBM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)
        d = self.backbone.num_features
        self.concept = nn.Sequential(nn.Linear(d, C), nn.Sigmoid())
        self.g = nn.Sequential(nn.Linear(C, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def forward(self, x):
        f = self.backbone(prep(x))
        c = self.concept(f)
        return c, self.g(c).squeeze(1)

    def mal_from_concepts(self, c):
        return self.g(c).squeeze(1)


def run():
    aurocs, intervs = [], []
    for fold in range(K):
        tr, va, te = fold_datasets(npz, cfg, fold)
        torch.manual_seed(4000 + fold); np.random.seed(4000 + fold)
        tl = DataLoader(tr, batch_size=16, shuffle=True, drop_last=True)
        vl = DataLoader(va, batch_size=32)
        m = ViTCBM().to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=1e-4)
        best, bs, pat = 0.0, None, 0
        for ep in range(25):
            m.train()
            for img, mask, con, reg, y in tl:
                img, con, y = augment(img.to(dev)), con.to(dev), y.to(dev)
                opt.zero_grad()
                chat, _ = m(img)
                loss = F.binary_cross_entropy_with_logits(m.mal_from_concepts(con), y.float()) \
                    + 0.5 * F.mse_loss(chat, con)
                loss.backward(); opt.step()
            m.eval(); ys, ps = [], []
            with torch.no_grad():
                for img, mask, con, reg, y in vl:
                    _, ml = m(img.to(dev)); ps.append(torch.sigmoid(ml).cpu().numpy()); ys.append(y.numpy())
            ys, ps = np.concatenate(ys), np.concatenate(ps)
            a = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else 0.5
            if a > best: best, bs, pat = a, copy.deepcopy(m.state_dict()), 0
            else:
                pat += 1
                if pat >= 6: break
        m.load_state_dict(bs)
        m.eval(); ys, ps, pt = [], [], []
        with torch.no_grad():
            for img, mask, con, reg, y in DataLoader(te, batch_size=32):
                img, con = img.to(dev), con.to(dev)
                _, ml = m(img); ps.append(torch.sigmoid(ml).cpu().numpy())
                pt.append(torch.sigmoid(m.mal_from_concepts(con)).cpu().numpy()); ys.append(y.numpy())
        ys = np.concatenate(ys)
        aurocs.append(float(roc_auc_score(ys, np.concatenate(ps))))
        intervs.append(float(roc_auc_score(ys, np.concatenate(pt))))
        print(f"  fold {fold}: AUROC={aurocs[-1]:.3f} interv={intervs[-1]:.3f}", flush=True)
    out = {"auroc_mean": round(float(np.mean(aurocs)), 3), "auroc_std": round(float(np.std(aurocs, ddof=1)), 3),
           "intervention_mean": round(float(np.mean(intervs)), 3), "folds": [round(x, 3) for x in aurocs]}
    json.dump(out, open("runs/transformer_cbm.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
