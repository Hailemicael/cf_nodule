"""Compute our faithful model's 5-fold accuracy (to compare with published accuracies)."""
import json, copy, yaml
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
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

accs,baccs,aurocs=[],[],[]
for fold in range(K):
    tr,va,te=fold_datasets(npz,cfg,fold)
    torch.manual_seed(4000+fold); np.random.seed(4000+fold)
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
    m.load_state_dict(bs)
    m.eval(); ys,ps=[],[]
    with torch.no_grad():
        for img,mask,con,reg,y in DataLoader(te,batch_size=32):
            ps.append(torch.sigmoid(m(img.to(dev))["malignancy"]).cpu().numpy()); ys.append(y.numpy())
    ys,ps=np.concatenate(ys),np.concatenate(ps); pred=(ps>0.5).astype(int)
    accs.append(accuracy_score(ys,pred)); aurocs.append(roc_auc_score(ys,ps))
    # balanced accuracy
    tp=((pred==1)&(ys==1)).sum()/max(1,(ys==1).sum()); tn=((pred==0)&(ys==0)).sum()/max(1,(ys==0).sum())
    baccs.append((tp+tn)/2)
    print(f"  fold {fold}: acc={accs[-1]:.3f} bacc={baccs[-1]:.3f} auroc={aurocs[-1]:.3f}",flush=True)
out={"accuracy_mean":round(float(np.mean(accs)),3),"accuracy_std":round(float(np.std(accs,ddof=1)),3),
     "balanced_acc_mean":round(float(np.mean(baccs)),3),"auroc_mean":round(float(np.mean(aurocs)),3)}
json.dump(out,open("runs/our_accuracy.json","w"),indent=2); print(json.dumps(out,indent=2))
