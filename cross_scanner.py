"""Cross-scanner generalization: leave-one-manufacturer-out. Train the faithful CBM
on all other CT vendors, test on the held-out vendor (a domain-shift / external-like
test within LIDC)."""
import json, copy, yaml
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from src.models.cf_nodule import CFNodule
from src.losses import dice_loss
from kfold import load_npz

dev="cuda" if torch.cuda.is_available() else "cpu"
cfg=yaml.safe_load(open("configs/default.yaml")); npz=load_npz(cfg)
pid=npz["pid"]; manu=npz["manufacturer"] if "manufacturer" in npz.files else np.array(["UNKNOWN"]*len(pid))

def vendor(s):
    s=str(s).upper()
    for v in ("SIEMENS","PHILIPS","TOSHIBA","GE"):
        if v in s: return v
    return "OTHER"
ven=np.array([vendor(x) for x in manu])

def ds(ix):
    t=lambda k: torch.tensor(npz[k][ix])
    return TensorDataset(t("image").float(),t("mask").float(),t("concept").float(),
                         t("region").float(),torch.tensor(npz["y"][ix]).long())

def aug(img,mask,reg):
    if torch.rand(1).item()<0.5: img,mask,reg=[torch.flip(t,[-1]) for t in (img,mask,reg)]
    if torch.rand(1).item()<0.5: img,mask,reg=[torch.flip(t,[-2]) for t in (img,mask,reg)]
    k=int(torch.randint(0,4,(1,)).item())
    if k: img,mask,reg=[torch.rot90(t,k,[-2,-1]) for t in (img,mask,reg)]
    return img,mask,reg

def train(tr,va,seed=4242):
    torch.manual_seed(seed); np.random.seed(seed)
    tl=DataLoader(tr,batch_size=16,shuffle=True,drop_last=True); vl=DataLoader(va,batch_size=16)
    m=CFNodule(n_concepts=7,base=cfg["model"]["base_channels"],depth=cfg["model"]["depth"],
               concept_bottleneck=True,mal_concept_hidden=128).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=cfg["train"]["lr"],weight_decay=cfg["train"]["weight_decay"])
    best,bs,pat=0.0,None,0
    for ep in range(120):
        m.train()
        for img,mask,con,reg,y in tl:
            img,mask,reg=aug(img.to(dev),mask.to(dev),reg.to(dev)); con,y=con.to(dev),y.to(dev)
            opt.zero_grad(); out=m(img)
            L=F.binary_cross_entropy_with_logits(m.malignancy_from_concepts(con),y.float())+1.0*F.mse_loss(out["concepts"],con)+dice_loss(out["seg"],mask)
            L.backward(); opt.step()
        m.eval(); ys,ps=[],[]
        with torch.no_grad():
            for img,mask,con,reg,y in vl:
                ps.append(torch.sigmoid(m(img.to(dev))["malignancy"]).cpu().numpy()); ys.append(y.numpy())
        ys,ps=np.concatenate(ys),np.concatenate(ps); a=roc_auc_score(ys,ps) if len(np.unique(ys))>1 else 0.5
        if a>best: best,bs,pat=a,copy.deepcopy(m.state_dict()),0
        else:
            pat+=1
            if pat>=25: break
    m.load_state_dict(bs); return m

counts={v:int((ven==v).sum()) for v in set(ven)}
print("vendor nodule counts:",counts,flush=True)
results={}
rng=np.random.default_rng(7)
for v in sorted(counts,key=lambda k:-counts[k]):
    test_idx=np.where(ven==v)[0]
    if len(test_idx)<40 or len(np.unique(npz["y"][test_idx]))<2: continue
    train_pool=np.where(ven!=v)[0]
    # patient-level val split from training vendors
    tp=np.unique(pid[train_pool]); rng.shuffle(tp); nval=max(1,int(0.15*len(tp)))
    valp=set(tp[:nval].tolist())
    vidx=np.array([i for i in train_pool if pid[i] in valp])
    tidx=np.array([i for i in train_pool if pid[i] not in valp])
    m=train(ds(tidx),ds(vidx))
    m.eval(); ys,ps=[],[]
    with torch.no_grad():
        for img,mask,con,reg,y in DataLoader(ds(test_idx),batch_size=32):
            ps.append(torch.sigmoid(m(img.to(dev))["malignancy"]).cpu().numpy()); ys.append(y.numpy())
    ys,ps=np.concatenate(ys),np.concatenate(ps)
    auc=float(roc_auc_score(ys,ps))
    results[v]={"test_n":int(len(test_idx)),"pos":int(ys.sum()),"auroc":round(auc,3)}
    print(f"  held-out {v}: n={len(test_idx)} AUROC={auc:.3f}",flush=True)

vals=[r["auroc"] for r in results.values()]
out={"per_vendor":results,"mean_auroc":round(float(np.mean(vals)),3) if vals else None,
     "std_auroc":round(float(np.std(vals,ddof=1)),3) if len(vals)>1 else None}
json.dump(out,open("runs/cross_scanner.json","w"),indent=2); print(json.dumps(out,indent=2))
