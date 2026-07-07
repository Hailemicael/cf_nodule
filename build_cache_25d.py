"""Build a 2.5D cache: 3 adjacent axial slices per nodule (more context) -> (N,3,P,P)."""
import os, re, glob, yaml
import numpy as np
from src.data.lidc_dataset import _hu_normalize, _center_crop, _boundary_band, ATTR_MAX
import pylidc as pl
from pylidc.utils import consensus

cfg = yaml.safe_load(open("configs/default.yaml"))
P = cfg["data"]["patch_size"]; concepts = cfg["data"]["concepts"]
minrad = cfg["data"]["min_radiologists"]; mthr = cfg["data"]["malignancy_threshold"]
excl = cfg["data"]["exclude_indeterminate"]; root = cfg["data"]["root"]

have = {d for d in os.listdir(root) if re.fullmatch(r"LIDC-IDRI-\d+", d)}
scans = [s for s in pl.query(pl.Scan).all() if s.patient_id in have]
print(f"[info] {len(scans)} scans on disk")

rows = []; skipped = 0
for si, scan in enumerate(scans):
    try:
        vol = scan.to_volume()
    except Exception:
        skipped += 1; continue
    S = vol.shape[2]
    for nod in scan.cluster_annotations():
        if len(nod) < minrad:
            continue
        mal = np.mean([a.malignancy for a in nod])
        if excl and abs(mal - 3.0) < 1e-6:
            continue
        label = int(mal > mthr)
        cmask, cbbox, _ = consensus(nod, clevel=0.5, pad=[(P, P), (P, P), (0, 0)])
        z = int(np.argmax(cmask.sum(axis=(0, 1))))
        zc = cbbox[2].start + z
        chans = []
        for dz in (-1, 0, 1):
            zz = min(max(zc + dz, 0), S - 1)
            img2d = _center_crop(vol[cbbox[0], cbbox[1], zz], P)
            chans.append(_hu_normalize(img2d))
        img3 = np.stack(chans)                       # (3,P,P)
        m2d = _center_crop(cmask[:, :, z].astype(np.float32), P)
        ct = []
        for c in concepts:
            key = "internalStructure" if c == "internal_structure" else c
            ct.append(float(np.mean([getattr(a, key) for a in nod])) / ATTR_MAX.get(key, 5))
        rows.append(dict(image=img3.astype(np.float32), mask=m2d,
                         concept=np.array(ct, np.float32), region=_boundary_band(m2d),
                         y=label, pid=scan.patient_id))
    if (si + 1) % 100 == 0:
        print(f"  {si+1}/{len(scans)} scans, {len(rows)} nodules", flush=True)

out = os.path.join(cfg["data"]["cache"], "lidc_patches_25d.npz")
np.savez_compressed(out,
    image=np.stack([r["image"] for r in rows]),                  # (N,3,P,P)
    mask=np.stack([r["mask"] for r in rows])[:, None],
    concept=np.stack([r["concept"] for r in rows]),
    region=np.stack([r["region"] for r in rows])[:, None],
    y=np.array([r["y"] for r in rows]),
    pid=np.array([r["pid"] for r in rows]))
print(f"[ok] wrote {out}  ({len(rows)} nodules, {skipped} skipped)")
