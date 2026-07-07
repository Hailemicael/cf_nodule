"""2.5D (3-slice) + wider backbone (base 48) + 3-seed ensemble, 5-fold, full LIDC.
Reports ensembled AUROC for the black box and the CEM (with intervention)."""
import json, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from src.models.cf_nodule import conv_block

dev = "cuda" if torch.cuda.is_available() else "cpu"
BASE, DEPTH, C, EMB, K, ENS = 48, 4, 7, 16, 5, 3
FOLD_SEED = 7
npz = np.load("data/cache/lidc_patches_25d.npz", allow_pickle=True)
pid = npz["pid"]; print("nodules", len(npz["y"]), "img", npz["image"].shape, flush=True)


def folds():
    up = np.unique(pid); np.random.default_rng(FOLD_SEED).shuffle(up)
    parts = np.array_split(up, K)
    for f in range(K):
        test = set(parts[f].tolist()); rest = [p for p in up if p not in test]
        nval = max(1, int(0.15 * len(rest))); val = set(rest[:nval]); tr = set(rest[nval:])
        idx = lambda g: np.array([i for i, p in enumerate(pid) if p in g])
        yield idx(tr), idx(val), idx(test)


def ds(ix):
    t = lambda k: torch.tensor(npz[k][ix])
    return TensorDataset(t("image").float(), t("concept").float(), torch.tensor(npz["y"][ix]).long())


def aug(x):
    if torch.rand(1).item() < 0.5: x = torch.flip(x, [-1])
    if torch.rand(1).item() < 0.5: x = torch.flip(x, [-2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k: x = torch.rot90(x, k, [-2, -1])
    return x


class Net(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        chs = [BASE * (2 ** i) for i in range(DEPTH + 1)]
        self.enc = nn.ModuleList(); cin = 3
        for c in chs[:-1]:
            self.enc.append(conv_block(cin, c)); cin = c
        self.pool = nn.MaxPool2d(2); self.bottleneck = conv_block(chs[-2], chs[-1]); h = chs[-1]
        self.concept = nn.Sequential(nn.Linear(h, C), nn.Sigmoid())
        if kind == "blackbox":
            self.head = nn.Sequential(nn.Linear(h, 128), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(128, 1))
        else:
            self.pos = nn.Linear(h, C * EMB); self.neg = nn.Linear(h, C * EMB)
            self.scorer = nn.Linear(2 * EMB, 1)
            self.head = nn.Sequential(nn.Linear(C * EMB, 128), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(128, 1))

    def feat(self, x):
        for e in self.enc: x = self.pool(e(x))
        return F.adaptive_avg_pool2d(self.bottleneck(x), 1).flatten(1)

    def forward(self, x):
        h = self.feat(x); c = self.concept(h)
        if self.kind == "blackbox":
            return c, self.head(h).squeeze(1)
        cp = self.pos(h).view(-1, C, EMB); cn = self.neg(h).view(-1, C, EMB)
        p = torch.sigmoid(self.scorer(torch.cat([cp, cn], -1)).squeeze(-1))
        mix = p.unsqueeze(-1) * cp + (1 - p).unsqueeze(-1) * cn
        return p, self.head(mix.flatten(1)).squeeze(1)

    def interv(self, x, cval):
        h = self.feat(x); cp = self.pos(h).view(-1, C, EMB); cn = self.neg(h).view(-1, C, EMB)
        mix = cval.unsqueeze(-1) * cp + (1 - cval).unsqueeze(-1) * cn
        return self.head(mix.flatten(1)).squeeze(1)


def train_one(kind, tr, va, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tl = DataLoader(tr, batch_size=16, shuffle=True, drop_last=True); vl = DataLoader(va, batch_size=32)
    m = Net(kind).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-4)
    best, bs, pat = 0.0, None, 0
    for ep in range(100):
        m.train()
        for x, con, y in tl:
            x = aug(x.to(dev)); con, y = con.to(dev), y.to(dev)
            opt.zero_grad(); p, ml = m(x)
            L = F.binary_cross_entropy_with_logits(ml, y.float()) + 1.0 * F.mse_loss(p, con)
            L.backward(); opt.step()
        m.eval(); ys, ps = [], []
        with torch.no_grad():
            for x, con, y in vl:
                _, ml = m(x.to(dev)); ps.append(torch.sigmoid(ml).cpu().numpy()); ys.append(y.numpy())
        ys, ps = np.concatenate(ys), np.concatenate(ps); a = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else .5
        if a > best: best, bs, pat = a, copy.deepcopy(m.state_dict()), 0
        else:
            pat += 1
            if pat >= 20: break
    m.load_state_dict(bs); return m


res = {}
try:
    _prev = json.load(open("runs/ensemble_25d.json"))
    for _k, _v in _prev.items():
        if isinstance(_v, dict) and "auroc_mean" in _v:
            res[_k] = _v; print(f"[info] reusing saved {_k} result", flush=True)
except Exception:
    pass
for kind in ("blackbox", "cem"):
    if kind in res:                                    # already computed in a prior run
        continue
    aurocs, intervs = [], []
    for f, (tri, vai, tei) in enumerate(folds()):
        tr, va, te = ds(tri), ds(vai), ds(tei)
        probs, iprobs, ys = [], [], None
        for s in range(ENS):
            m = train_one(kind, tr, va, 100 * s + f); m.eval()
            pp, ip, yy = [], [], []
            with torch.no_grad():
                for x, con, y in DataLoader(te, batch_size=32):
                    x, con = x.to(dev), con.to(dev)
                    _, ml = m(x); pp.append(torch.sigmoid(ml).cpu().numpy())
                    if kind == "cem": ip.append(torch.sigmoid(m.interv(x, con)).cpu().numpy())
                    yy.append(y.numpy())
            probs.append(np.concatenate(pp)); ys = np.concatenate(yy)
            if kind == "cem": iprobs.append(np.concatenate(ip))
            del m; torch.cuda.empty_cache()              # free GPU between models
        pe = np.mean(probs, 0); aurocs.append(float(roc_auc_score(ys, pe)))
        if kind == "cem": intervs.append(float(roc_auc_score(ys, np.mean(iprobs, 0))))
        print(f"  [{kind}] fold {f}: ens AUROC={aurocs[-1]:.3f}", flush=True)
        # incremental save so a crash never loses progress
        partial = dict(res); partial[kind] = {"auroc_folds_so_far": [round(x, 3) for x in aurocs]}
        json.dump(partial, open("runs/ensemble_25d.json", "w"), indent=2)
    res[kind] = {"auroc_mean": round(float(np.mean(aurocs)), 3), "auroc_std": round(float(np.std(aurocs, ddof=1)), 3),
                 "folds": [round(x, 3) for x in aurocs]}
    if intervs: res[kind]["intervention_mean"] = round(float(np.mean(intervs)), 3)
    json.dump(res, open("runs/ensemble_25d.json", "w"), indent=2)
    print(json.dumps({kind: res[kind]}, indent=2), flush=True)

json.dump(res, open("runs/ensemble_25d.json", "w"), indent=2)
print("FINAL", json.dumps(res, indent=2))
