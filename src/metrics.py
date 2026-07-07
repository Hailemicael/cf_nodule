"""
Evaluation metrics for CF-Nodule.

These REPLACE "overlap with the lesion mask" as the measure of explanation quality:
  - insertion / deletion AUC (intervention faithfulness)
  - threshold-free AUC-IoU (localization descriptor, vs lesion AND concept band)
  - concept-alignment (AUC-IoU vs concept band)
  - Concept-Faithfulness Score CFS = faithfulness * concept_alignment
  - selective prediction (risk-coverage) + bootstrap confidence intervals
"""
import numpy as np
import torch


def _trapz(y):
    x = np.linspace(0, 1, len(y))
    integ = getattr(np, "trapezoid", np.trapz)
    return float(integ(y, x))


@torch.no_grad()
def insertion_deletion(model, x, cam, steps=25, baseline=0.0):
    """x: 1x1xPxP tensor, cam: 1x1xPxP in [0,1]. Returns (ins_auc, del_auc)."""
    dev = x.device
    order = torch.argsort(cam.view(-1), descending=True)
    flat = x.view(-1).clone()
    ins = torch.full_like(flat, baseline)
    dele = flat.clone()
    chunk = max(1, len(order) // steps)
    ins_curve, del_curve = [], []
    def p(v):
        return torch.sigmoid(model(v.view_as(x))["malignancy"]).item()
    ins_curve.append(p(ins)); del_curve.append(p(dele))
    for s in range(steps):
        idx = order[s * chunk:(s + 1) * chunk]
        ins[idx] = flat[idx]; dele[idx] = baseline
        ins_curve.append(p(ins)); del_curve.append(p(dele))
    return _trapz(ins_curve), _trapz(del_curve), ins_curve, del_curve


def auc_iou(cam, mask, n_th=25):
    cam = cam.squeeze().cpu().numpy() if torch.is_tensor(cam) else np.asarray(cam).squeeze()
    mask = mask.squeeze().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask).squeeze()
    m = mask > 0.5
    ious = []
    for t in np.linspace(0.05, 0.95, n_th):
        b = cam >= t
        u = (b | m).sum()
        ious.append((b & m).sum() / u if u > 0 else 0.0)
    return float(np.mean(ious))


def faithfulness_score(ins_auc, del_auc):
    """Map insertion/deletion to a single [0,1] score (higher=better)."""
    return float(np.clip(0.5 * (ins_auc + (1 - del_auc)), 0, 1))


def cfs(faith, align):
    return float(faith * align)


def risk_coverage(y, prob, reliability, thr=0.5):
    y = np.asarray(y); prob = np.asarray(prob); reliability = np.asarray(reliability)
    order = np.argsort(reliability)[::-1]
    y_o = y[order]; pred = (prob[order] > thr).astype(int)
    cov, acc = [], []
    for c in range(max(5, len(y) // 12), len(y) + 1):
        cov.append(c / len(y)); acc.append(float(np.mean(pred[:c] == y_o[:c])))
    return np.array(cov), np.array(acc)


def best_threshold(y, p):
    y = np.asarray(y); p = np.asarray(p)
    ths = np.unique(np.round(p, 3)); best, bt = -1, 0.5
    for t in ths:
        a = np.mean((p > t).astype(int) == y)
        if a > best: best, bt = a, t
    return float(bt)


def bootstrap_ci(values, n=1000, alpha=0.05, seed=0):
    """95% CI of the mean via bootstrap (for small-sample rigor)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(n)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap_delta(a, b, n=5000, alpha=0.05, seed=0):
    """Paired bootstrap of the per-item mean difference mean(a - b).

    Both models are scored on the SAME test nodules, so the per-nodule difference
    is paired -- this removes between-nodule variance and is far more powerful than
    comparing two independent means' CIs. Returns (delta, lo, hi, p_two_sided).
    """
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b
    rng = np.random.default_rng(seed)
    idx = np.arange(len(d))
    boots = np.array([d[rng.choice(idx, len(d), replace=True)].mean() for _ in range(n)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())   # two-sided bootstrap p
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def auroc_paired_bootstrap(y, p1, p2, n=5000, alpha=0.05, seed=0):
    """Paired bootstrap of AUROC(p1) - AUROC(p2) on the same labels y.

    DeLong-style paired comparison via case resampling. Returns
    (auc1, auc2, delta, lo, hi, p_two_sided).
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p1 = np.asarray(p1); p2 = np.asarray(p2)
    auc1 = roc_auc_score(y, p1) if len(np.unique(y)) > 1 else 0.5
    auc2 = roc_auc_score(y, p2) if len(np.unique(y)) > 1 else 0.5
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); diffs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        diffs.append(roc_auc_score(y[s], p1[s]) - roc_auc_score(y[s], p2[s]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(auc1), float(auc2), float(auc1 - auc2), float(lo), float(hi), float(min(p, 1.0))


def wilcoxon_p(a, b):
    """Wilcoxon signed-rank p-value for paired a vs b (non-parametric)."""
    try:
        from scipy.stats import wilcoxon
        a = np.asarray(a, float); b = np.asarray(b, float)
        if np.allclose(a, b):
            return 1.0
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        return float("nan")
