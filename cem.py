"""Concept Embedding Model (CEM, Espinosa-Zarlenga et al. 2022) on full LIDC, 5-fold.
Each concept gets a positive and negative embedding; the predicted concept
probability mixes them. The malignancy head reads the mixed concept embeddings, so
the model keeps test-time intervention while regaining capacity to close the
CBM accuracy gap. Reports AUROC (predicted concepts) and intervention AUROC (true c).
"""
import json, copy, yaml
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from src.models.cf_nodule import conv_block
from kfold import load_npz, fold_datasets, K

dev="cuda" if torch.cuda.is_available() else "cpu"
cfg=yaml.safe_load(open("configs/default.yaml")); npz=load_npz(cfg)
C=7; EMB=16

def aug(img):
    if torch.rand(1).item()<0.5: img=torch.flip(img,[-1])
    if torch.rand(1).item()<0.5: img=torch.flip(img,[-2])
    k=int(torch.randint(0,4,(1,)).item())
    if k: img=torch.rot90(img,k,[-2,-1])
    return img

class CEM(nn.Module):
    def __init__(self, base=32, depth=4):
        super().__init__()
        chs=[base*(2**i) for i in range(depth+1)]
        self.enc=nn.ModuleList(); cin=1
        for c in chs[:-1]:
            self.enc.append(conv_block(cin,c)); cin=c
        self.pool=nn.MaxPool2d(2); self.bottleneck=conv_block(chs[-2],chs[-1])
        h=chs[-1]
        self.pos=nn.Linear(h,C*EMB); self.neg=nn.Linear(h,C*EMB)
        self.scorer=nn.Linear(2*EMB,1)
        self.head=nn.Sequential(nn.Linear(C*EMB,128),nn.ReLU(inplace=True),nn.Dropout(0.3),nn.Linear(128,1))
    def embed(self,x):
        hh=x
        for e in self.enc: hh=self.pool(e(hh))
        hh=self.bottleneck(hh); h=F.adaptive_avg_pool2d(hh,1).flatten(1)
        cp=self.pos(h).view(-1,C,EMB); cn=self.neg(h).view(-1,C,EMB)
        p=torch.sigmoid(self.scorer(torch.cat([cp,cn],-1)).squeeze(-1))  # B,C
        return cp,cn,p
    def forward(self,x):
        cp,cn,p=self.embed(x)
        mix=p.unsqueeze(-1)*cp+(1-p).unsqueeze(-1)*cn
        return p, self.head(mix.flatten(1)).squeeze(1)
    def from_concepts(self,x,c):  # intervention: use given concept values c (B,C)
        cp,cn,_=self.embed(x)
        mix=c.unsqueeze(-1)*cp+(1-c).unsqueeze(-1)*cn
        return self.head(mix.flatten(1)).squeeze(1)

def train(tr,va,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tl=DataLoader(tr,batch_size=16,shuffle=True,drop_last=True); vl=DataLoader(va,batch_size=16)
    m=CEM(base=cfg["model"]["base_channels"],depth=cfg["model"]["depth"]).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=cfg["train"]["lr"],weight_decay=cfg["train"]["weight_decay"])
    best,bs,pat=0.0,None,0
    for ep in range(120):
        m.train()
        for img,mask,con,reg,y in tl:
            img=aug(img.to(dev)); con,y=con.to(dev),y.to(dev)
            opt.zero_grad(); p,ml=m(img)
            L=F.binary_cross_entropy_with_logits(ml,y.float())+1.0*F.mse_loss(p,con)
            L.backward(); opt.step()
        m.eval(); ys,ps=[],[]
        with torch.no_grad():
            for img,mask,con,reg,y in vl:
                _,ml=m(img.to(dev)); ps.append(torch.sigmoid(ml).cpu().numpy()); ys.append(y.numpy())
        ys,ps=np.concatenate(ys),np.concatenate(ps); a=roc_auc_score(ys,ps) if len(np.unique(ys))>1 else 0.5
        if a>best: best,bs,pat=a,copy.deepcopy(m.state_dict()),0
        else:
            pat+=1
            if pat>=25: break
    m.load_state_dict(bs); return m

aurocs,intervs=[],[]
for fold in range(K):
    tr,va,te=fold_datasets(npz,cfg,fold)
    m=train(tr,va,4000+fold); m.eval()
    ys,ps,pt=[],[],[]
    with torch.no_grad():
        for img,mask,con,reg,y in DataLoader(te,batch_size=32):
            img,con=img.to(dev),con.to(dev)
            _,ml=m(img); ps.append(torch.sigmoid(ml).cpu().numpy())
            pt.append(torch.sigmoid(m.from_concepts(img,con)).cpu().numpy()); ys.append(y.numpy())
    ys=np.concatenate(ys)
    aurocs.append(float(roc_auc_score(ys,np.concatenate(ps))))
    intervs.append(float(roc_auc_score(ys,np.concatenate(pt))))
    print(f"  fold {fold}: AUROC={aurocs[-1]:.3f} interv={intervs[-1]:.3f}",flush=True)
out={"auroc_mean":round(float(np.mean(aurocs)),3),"auroc_std":round(float(np.std(aurocs,ddof=1)),3),
     "intervention_mean":round(float(np.mean(intervs)),3),"folds":[round(x,3) for x in aurocs]}
json.dump(out,open("runs/cem.json","w"),indent=2); print(json.dumps(out,indent=2))
