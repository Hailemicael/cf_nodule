"""
Make a SYNTHETIC LIDC-format patch cache so the FULL CF-Nodule pipeline
(src.train / src.evaluate) can run end-to-end without the gated 125 GB DICOM data.

This writes exactly the .npz schema that src/data/lidc_dataset.py:build_cache would
produce from real LIDC-IDRI, so train.py/evaluate.py consume it unchanged:

    image   : N x 1 x P x P  float32   (HU-normalized CT-like patch)
    mask     : N x 1 x P x P  float32   (consensus nodule mask)
    concept  : N x C          float32   (LIDC semantic ratings, normalized to [0,1])
    region   : N x 1 x P x P  float32   (boundary band for concept-alignment scoring)
    y        : N              int64     (malignant=1 / benign=0)
    pid      : N              <U..>      (patient id -> patient-level split)

HONESTY: the pixels are synthetic, not real CT. Malignancy is driven by the
spiculation/lobulation concepts placed on the nodule rim, so the pipeline and the
CFA regularizer have a real (if toy) signal to learn and explain. Numbers from this
illustrate the *mechanism / that the code works*, NOT clinical performance.

    python demo/make_synth_cache.py --config configs/quick.yaml --patients 80
"""
import os, argparse, yaml
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion

# concept order must match cfg["data"]["concepts"]; we drive y from spic/lob.
def _concepts_for(driver, concepts, rng):
    """Return a rating in [0,1] per requested concept; spiculation/lobulation
    track the malignant driver, the rest are weak/noisy (like real LIDC)."""
    table = {
        "spiculation": np.clip(driver + rng.normal(0, 0.10), 0, 1),
        "lobulation":  np.clip(driver + rng.normal(0, 0.12), 0, 1),
        "margin":      np.clip(driver * 0.6 + rng.normal(0, 0.15), 0, 1),
        "subtlety":    np.clip(0.5 + rng.normal(0, 0.18), 0, 1),
        "calcification": np.clip(0.7 - 0.3 * driver + rng.normal(0, 0.15), 0, 1),
        "sphericity":  np.clip(0.6 - 0.2 * driver + rng.normal(0, 0.15), 0, 1),
        "texture":     np.clip(0.6 + rng.normal(0, 0.15), 0, 1),
    }
    return np.array([table.get(c, np.clip(rng.uniform(), 0, 1)) for c in concepts],
                    dtype=np.float32)


def _sample(P, driver, rng):
    """One synthetic CT-like nodule patch + its consensus mask."""
    img = gaussian_filter(rng.normal(0.35, 0.05, (P, P)), sigma=2.0)   # lung texture
    yy, xx = np.mgrid[0:P, 0:P]
    cy, cx = P / 2 + rng.normal(0, 1.5), P / 2 + rng.normal(0, 1.5)
    r = 0.16 * P + rng.normal(0, 0.6)
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    nodule = (dist <= r).astype(np.float32)
    img = img + 0.5 * nodule * gaussian_filter(np.ones((P, P)), 1.0)

    # spiculation = radial spikes on the rim; strength scales with malignant driver
    ang = np.arctan2(yy - cy, xx - cx)
    n_spk = int(4 + 8 * driver)
    spic = (np.cos(ang * n_spk) > 0.6) & (dist > r - 1) & (dist < r + 4 * driver + 1)
    lob = (np.cos(ang * 3) > 0.3) & (np.abs(dist - r) < 1.0 + 1.5 * driver)
    rim = (spic | lob).astype(np.float32)
    img = img + 0.45 * rim                                  # signal sits on the rim/band
    img = np.clip(img + rng.normal(0, 0.03, (P, P)), 0, 1.5).astype(np.float32)
    return img, nodule


def _boundary_band(mask, width=3):
    m = mask > 0.5
    band = binary_dilation(m, iterations=width) & ~binary_erosion(m, iterations=width)
    return band.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/quick.yaml")
    ap.add_argument("--patients", type=int, default=80)
    ap.add_argument("--per-patient", type=int, default=4)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    P = cfg["data"]["patch_size"]
    concepts = cfg["data"]["concepts"]
    cache = cfg["data"]["cache"]
    os.makedirs(cache, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])

    images, masks, regions, conwx, ys, pids = [], [], [], [], [], []
    for p in range(a.patients):
        pid = f"SYN-{p:04d}"
        for _ in range(a.per_patient):
            driver = rng.uniform(0, 1)
            y = int(driver + rng.normal(0, 0.12) > 0.5)
            img, nod = _sample(P, driver, rng)
            images.append(img[None]); masks.append(nod[None])
            regions.append(_boundary_band(nod)[None])
            conwx.append(_concepts_for(driver, concepts, rng))
            ys.append(y); pids.append(pid)

    out = os.path.join(cache, "lidc_patches.npz")
    np.savez_compressed(
        out,
        image=np.stack(images).astype(np.float32),
        mask=np.stack(masks).astype(np.float32),
        concept=np.stack(conwx).astype(np.float32),
        region=np.stack(regions).astype(np.float32),
        y=np.array(ys, dtype=np.int64),
        pid=np.array(pids),
    )
    n = len(ys)
    print(f"[ok] wrote {out}")
    print(f"     {n} synthetic nodules, {a.patients} patients, "
          f"pos_rate={np.mean(ys):.2f}, patch={P}, concepts={len(concepts)}")


if __name__ == "__main__":
    main()
