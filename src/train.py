"""
Train CF-Nodule on the LIDC patch cache.

    python -m src.train --config configs/default.yaml --cfa on    # proposed
    python -m src.train --config configs/default.yaml --cfa off   # ablation baseline

Saves the best (val malignancy AUROC) checkpoint to runs/<tag>/best.pt
"""
import os, argparse, yaml, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.models.cf_nodule import CFNodule
from src.losses import total_loss
from src.data.lidc_dataset import make_datasets


def evaluate_auroc(model, loader, dev):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for img, mask, concept, region, y in loader:
            out = model(img.to(dev))
            ps.append(torch.sigmoid(out["malignancy"]).cpu().numpy()); ys.append(y.numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    return roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cfa", choices=["on", "off"], default="on")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.cfa == "off":
        cfg["train"]["w_cfa"] = 0.0
    tag = a.tag or f"cfnodule_cfa-{a.cfa}"
    outdir = os.path.join("runs", tag); os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={dev}  cfa={a.cfa}")

    tr, va, te = make_datasets(cfg)
    bs = cfg["train"]["batch_size"]
    tl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)
    vl = DataLoader(va, batch_size=bs)
    n_concepts = len(cfg["data"]["concepts"])
    model = CFNodule(n_concepts=n_concepts,
                     base=cfg["model"]["base_channels"],
                     depth=cfg["model"]["depth"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and dev == "cuda")

    best, patience = 0.0, 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train(); t0 = time.time(); running = 0.0
        for batch in tl:
            batch = [b.to(dev) for b in batch]
            img = batch[0]
            opt.zero_grad()
            # CFA needs graph through grad-cam, so autocast is disabled for that path
            out = model(img)
            loss, logs = total_loss(out, batch, model, img, cfg, epoch)
            loss.backward(); opt.step()
            running += float(loss)
        auroc = evaluate_auroc(model, vl, dev)
        print(f"epoch {epoch:03d}  loss={running/len(tl):.3f}  val_AUROC={auroc:.3f}  "
              f"{logs}  ({time.time()-t0:.1f}s)")
        if auroc > best:
            best, patience = auroc, 0
            torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(outdir, "best.pt"))
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                print("[info] early stop"); break
    print(f"[ok] best val AUROC={best:.3f}  -> {outdir}/best.pt")


if __name__ == "__main__":
    main()
