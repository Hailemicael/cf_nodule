"""Faithful CBM on FULL LIDC, 5-fold: AUROC, intervention, plus pooled test
predictions for selective prediction (risk-coverage) and calibration (ECE)."""
import json, copy, yaml
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.models.cf_nodule import CFNodule
from src.losses import dice_loss
from kfold import load_npz, fold_datasets, K

dev="cuda" if torch.cuda.is_available() else "cpu"
cfg=yaml.safe_load(open("configs/default.yaml")); npz=load_npz(cfg)

def aug(img,mask,reg):
    if torch.rand(1).item()<0.5: img,mask,reg=[torch.flip(t,[-1]) for t in (img,mask,reg)]
    if torch.rand(1).item()<0.5: img,mask,reg=[torch.flip(t,[-2]) for t in (img,mask,reg)]
    k=int(torch.randint(0,4,(1,)).item())
    if k: img,mask,reg=[torch.rot90(t,k,[-2,-1]) for t in (img,mask,reg)]
    return img,mask,reg

def train(tr,va,seed):
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

aurocs,intervs=[],[]; P,Y=[],[]
for fold in range(K):
    tr,va,te=fold_datasets(npz,cfg,fold)
    m=train(tr,va,4000+fold); m.eval()
    ys,ps,pt=[],[],[]
    with torch.no_grad():
        for img,mask,con,reg,y in DataLoader(te,batch_size=32):
            img,con=img.to(dev),con.to(dev)
            ps.append(torch.sigmoid(m(img)["malignancy"]).cpu().numpy())
            pt.append(torch.sigmoid(m.malignancy_from_concepts(con)).cpu().numpy()); ys.append(y.numpy())
    ys=np.concatenate(ys); ps=np.concatenate(ps); pt=np.concatenate(pt)
    aurocs.append(float(roc_auc_score(ys,ps))); intervs.append(float(roc_auc_score(ys,pt)))
    P.append(ps); Y.append(ys)
    print(f"  fold {fold}: AUROC={aurocs[-1]:.3f} interv={intervs[-1]:.3f}",flush=True)

P=np.concatenate(P); Y=np.concatenate(Y)
# ECE (10 bins)
bins=np.linspace(0,1,11); ece=0.0
for i in range(10):
    lo,hi=bins[i],bins[i+1]; m_=(P>=lo)&(P<hi) if i<9 else (P>=lo)&(P<=hi)
    if m_.sum()>0: ece+=abs(P[m_].mean()-Y[m_].mean())*m_.mean()
# risk-coverage (abstain on least-confident |p-0.5|)
conf=np.abs(P-0.5); order=np.argsort(conf)[::-1]; yo=Y[order]; pred=(P[order]>0.5).astype(int)
covs=np.arange(max(20,len(Y)//20),len(Y)+1); cov=covs/len(Y)
acc=np.array([ (pred[:c]==yo[:c]).mean() for c in covs ])
plt.figure(figsize=(6.2,4.0)); plt.plot(cov,acc,color="tab:green")
plt.xlabel("coverage (fraction auto-decided)"); plt.ylabel("accuracy on covered cases")
plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("runs/risk_coverage.png",dpi=150,bbox_inches="tight")

out=dict(auroc_mean=round(float(np.mean(aurocs)),3),auroc_std=round(float(np.std(aurocs,ddof=1)),3),
         intervention_mean=round(float(np.mean(intervs)),3),
         acc_full_coverage=round(float(accuracy_score(Y,(P>0.5).astype(int))),3),
         acc_at_80pct_coverage=round(float(acc[np.argmin(np.abs(cov-0.8))]),3),
         ECE=round(float(ece),3), n=int(len(Y)), folds=[round(x,3) for x in aurocs])
json.dump(out,open("runs/final_main.json","w"),indent=2); print(json.dumps(out,indent=2))
