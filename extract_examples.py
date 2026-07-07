"""Pull REAL high-resolution CT nodule crops straight from the downloaded LIDC DICOM
volumes (via pylidc) and lay them out. These are actual dataset images, not rendered."""
import os, re, configparser as _cp
import numpy as np
if not hasattr(_cp, "SafeConfigParser"):
    _cp.SafeConfigParser = _cp.ConfigParser
for _a, _t in {"int": int, "float": float, "bool": bool}.items():
    if not hasattr(np, _a):
        setattr(np, _a, _t)
import pylidc as pl
from pylidc.utils import consensus
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = 96                       # display crop size (bigger than the 64 model input, for clarity)
WL, WW = -600, 1500          # lung window
lo, hi = WL - WW / 2, WL + WW / 2


def center_crop(a, P):
    h, w = a.shape
    if h < P or w < P:
        a = np.pad(a, ((max(0, (P - h + 1) // 2),) * 2, (max(0, (P - w + 1) // 2),) * 2))
        h, w = a.shape
    y0, x0 = (h - P) // 2, (w - P) // 2
    return a[y0:y0 + P, x0:x0 + P]


root = "data/LIDC-IDRI"
have = {d for d in os.listdir(root) if re.fullmatch(r"LIDC-IDRI-\d+", d)}
scans = [s for s in pl.query(pl.Scan).all() if s.patient_id in have]

benign, malig = [], []
for s in scans:
    if len(benign) >= 3 and len(malig) >= 3:
        break
    try:
        vol = s.to_volume()
    except Exception:
        continue
    for nod in s.cluster_annotations():
        if len(nod) < 3:
            continue
        mal = float(np.mean([a.malignancy for a in nod]))
        if abs(mal - 3) < 1e-6:
            continue
        cmask, cbbox, _ = consensus(nod, clevel=0.5, pad=[(P, P), (P, P), (0, 0)])
        z = int(np.argmax(cmask.sum(axis=(0, 1))))
        img = center_crop(vol[cbbox[0], cbbox[1], cbbox[2].start + z].astype(float), P)
        m = center_crop(cmask[:, :, z].astype(float), P)
        disp = np.clip((img - lo) / (hi - lo), 0, 1)
        sp = np.mean([a.spiculation for a in nod]) / 5
        lobv = np.mean([a.lobulation for a in nod]) / 5
        mg = np.mean([a.margin for a in nod]) / 5
        rec = (disp, m, sp, lobv, mg)
        tgt = malig if mal > 3 else benign
        if len(tgt) < 3:
            tgt.append(rec)

idx = benign[:3] + malig[:3]
labels = ["benign"] * len(benign[:3]) + ["malignant"] * len(malig[:3])
fig, ax = plt.subplots(1, 6, figsize=(13, 2.8))
for a, (disp, m, sp, lobv, mg), lab in zip(ax, idx, labels):
    a.imshow(disp, cmap="gray")
    a.contour(m, levels=[0.5], colors="lime", linewidths=1.0)
    a.set_title(lab, fontsize=11, color=("firebrick" if lab == "malignant" else "seagreen"))
    a.set_xlabel(f"sp {sp:.2f}  lo {lobv:.2f}  mg {mg:.2f}", fontsize=8)
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig("runs/examples.png", dpi=150, bbox_inches="tight")
print(f"[ok] examples.png  ({len(benign)} benign, {len(malig)} malignant real CT crops)")
