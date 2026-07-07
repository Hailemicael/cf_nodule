"""
Losses for CF-Nodule, including the novel Concept-Faithfulness Alignment (CFA) term.

Total = w_seg*Dice + w_concept*MSE + w_mal*BCE + w_cfa*CFA

CFA has two coupled goals on the malignancy Grad-CAM map `cam`:
  (1) faithfulness  : the highlighted evidence should actually drive the decision
                      -> we use a differentiable deletion/insertion surrogate
                         (masking high-cam pixels must drop the malignancy logit).
  (2) concept-align : that faithful evidence should sit on the clinically-relevant
                      concept region (spiculation/lobulation/margin band).
"""
import torch
import torch.nn.functional as F


def dice_loss(pred_logits, target, eps=1.0):
    p = torch.sigmoid(pred_logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def concept_loss(pred, target):
    return F.mse_loss(pred, target)


def malignancy_loss(logits, y):
    return F.binary_cross_entropy_with_logits(logits, y.float())


def cfa_loss(model, x, cam, concept_region, baseline=0.0, topk=0.2,
             w_faith=1.0, w_align=1.0):
    """Concept-Faithfulness Alignment with separately weighted terms.

    faithfulness_term: deleting the top-k cam pixels should LOWER P(malignant);
                       keeping only them should preserve it (insertion).
    align_term       : 1 - (mass of cam inside concept_region) -> push evidence
                       onto the clinically meaningful band.
    w_faith / w_align expose the two terms for the ablation showing they conflict
    (boosting alignment degrades faithfulness under low-resolution Grad-CAM).
    Returns (weighted_loss, raw_faith, raw_align, logs).
    """
    B = x.shape[0]
    flat = cam.view(B, -1)
    k = max(1, int(topk * flat.shape[1]))
    thresh = flat.topk(k, dim=1).values[:, -1:].view(B, 1, 1, 1)
    high = (cam >= thresh).float()

    base_logit = model(x)["malignancy"]
    deleted = x * (1 - high) + baseline * high          # remove salient evidence
    kept = x * high + baseline * (1 - high)             # keep only salient evidence
    del_logit = model(deleted)["malignancy"]
    keep_logit = model(kept)["malignancy"]

    # faithful: base ~ keep (insertion high) and del << base (deletion low)
    faith = (F.relu(del_logit - base_logit + 0.0).mean()           # deletion should drop
             + F.relu(base_logit - keep_logit).mean())             # insertion should hold
    # alignment: cam mass should fall inside the concept band
    mass_in = (cam * concept_region).sum(dim=(1, 2, 3))
    mass_all = cam.sum(dim=(1, 2, 3)) + 1e-6
    align = (1 - mass_in / mass_all).mean()
    loss = w_faith * faith + w_align * align
    return loss, float(faith), float(align), dict(faith=float(faith), align=float(align))


def total_loss(out, batch, model, x, cfg, epoch):
    img, mask, concept, region, y = batch
    L = (cfg["train"]["w_seg"] * dice_loss(out["seg"], mask)
         + cfg["train"]["w_concept"] * concept_loss(out["concepts"], concept)
         + cfg["train"]["w_malignancy"] * malignancy_loss(out["malignancy"], y))
    logs = {}
    tr = cfg["train"]
    if epoch >= tr["cfa_warmup_epochs"] and tr["w_cfa"] > 0:
        cam = model.gradcam_malignancy(x)
        cfa, _, _, logs = cfa_loss(model, x, cam, region,
                                   w_faith=tr.get("cfa_w_faith", 1.0),
                                   w_align=tr.get("cfa_w_align", 1.0))
        L = L + tr["w_cfa"] * cfa
    return L, logs
