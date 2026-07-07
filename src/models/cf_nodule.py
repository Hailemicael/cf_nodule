"""
CF-Nodule model: a U-Net backbone with three heads on the shared encoder/decoder.

    segmentation head : 1-channel nodule mask  (reuses your existing GAN/U-Net seg)
    concept heads     : C scalar regressors for LIDC semantic attributes
    malignancy head   : 1 logit for malignant vs benign  (the decision we explain)

The malignancy head pools the deepest encoder features, which lets Grad-CAM /
gradient attribution be computed w.r.t. that feature map for the CFA regularizer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class CFNodule(nn.Module):
    def __init__(self, n_concepts=7, base=32, depth=4, concept_bottleneck=False,
                 mal_concept_hidden=32, mal_mode=None):
        super().__init__()
        self.depth = depth
        self.concept_bottleneck = concept_bottleneck
        self.n_concepts = n_concepts
        self.mal_concept_hidden = mal_concept_hidden
        # malignancy pathway: 'features' (black-box / CNN classifier),
        # 'concepts' (concept bottleneck), or 'augmented' (HSCNN-style: features +
        # predicted concepts). Defaults from concept_bottleneck for back-compat.
        self.mal_mode = mal_mode or ("concepts" if concept_bottleneck else "features")
        chs = [base * (2 ** i) for i in range(depth + 1)]   # e.g. 32,64,128,256,512
        # encoder
        self.enc = nn.ModuleList()
        cin = 1
        for c in chs[:-1]:
            self.enc.append(conv_block(cin, c)); cin = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(chs[-2], chs[-1])
        # decoder
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.up.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2))
            self.dec.append(conv_block(chs[i + 1], chs[i]))
        # heads
        self.seg_head = nn.Conv2d(chs[0], 1, 1)
        self.concept_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(chs[-1], 128), nn.ReLU(inplace=True),
            nn.Linear(128, n_concepts), nn.Sigmoid())          # ratings in [0,1]
        # Malignancy head reads a HIGHER-RESOLUTION decoder feature (input/4, e.g.
        # 16x16) rather than the 4x4 bottleneck, so Grad-CAM taken w.r.t. it is
        # fine enough to localize the thin concept band -- the fix for the
        # faithfulness/alignment failure of the bottleneck-CAM version.
        self.mal_ch = chs[2] if depth >= 2 else chs[-1]        # channels at input/4
        self.mal_pool = nn.AdaptiveAvgPool2d(1)
        self.mal_head = nn.Sequential(nn.Flatten(),
                                      nn.Linear(self.mal_ch, 128), nn.ReLU(inplace=True),
                                      nn.Dropout(0.3), nn.Linear(128, 1))
        # Concept-bottleneck path: malignancy is a function of the predicted concepts
        # ONLY (features -> concepts -> malignancy). Interpretable + faithful by
        # construction, and supports concept intervention.
        if mal_concept_hidden and mal_concept_hidden > 0:
            self.mal_from_concept = nn.Sequential(
                nn.Linear(n_concepts, mal_concept_hidden), nn.ReLU(inplace=True),
                nn.Linear(mal_concept_hidden, 1))
        else:                                                  # linear (max interpretable)
            self.mal_from_concept = nn.Linear(n_concepts, 1)
        # HSCNN-style augmented head: malignancy from [pooled features ; concepts]
        self.mal_aug = nn.Sequential(
            nn.Linear(self.mal_ch + n_concepts, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, 1))
        self._feat = None       # cached bottleneck features (concepts)
        self._mal_feat = None   # cached high-res decoder feature (malignancy + CAM)

    def forward(self, x, return_features=False):
        skips = []
        h = x
        for enc in self.enc:
            h = enc(h); skips.append(h); h = self.pool(h)
        feat = self.bottleneck(h)        # deepest features
        self._feat = feat
        # decode for segmentation; capture the input/4-resolution feature for malignancy
        target_hw = x.shape[-1] // 4
        self._mal_feat = None
        d = feat
        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            d = up(d)
            skip = skips[-(i + 1)]
            if d.shape[-2:] != skip.shape[-2:]:
                d = F.interpolate(d, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            d = dec(torch.cat([d, skip], dim=1))
            if d.shape[-1] == target_hw:
                self._mal_feat = d                       # high-res feature on decode path
        if self._mal_feat is None:
            self._mal_feat = feat                        # fallback (shallow nets)
        seg = self.seg_head(d)
        concepts = self.concept_head(feat)
        pooled = self.mal_pool(self._mal_feat).flatten(1)
        if self.mal_mode == "concepts":
            mal = self.mal_from_concept(concepts).squeeze(1)   # features->concepts->malignancy
        elif self.mal_mode == "augmented":
            mal = self.mal_aug(torch.cat([pooled, concepts], dim=1)).squeeze(1)  # HSCNN-style
        else:                                                  # 'features' (black-box)
            mal = self.mal_head(pooled).squeeze(1)
        out = dict(seg=seg, concepts=concepts, malignancy=mal,
                   features=feat, mal_features=self._mal_feat)
        return out if return_features else {k: out[k] for k in ("seg", "concepts", "malignancy")}

    def malignancy_from_concepts(self, concepts):
        """Predict malignancy from GIVEN concept values (for concept intervention)."""
        return self.mal_from_concept(concepts).squeeze(1)

    def gradcam_malignancy(self, x):
        """Grad-CAM for the malignancy logit w.r.t. the high-res decoder feature,
        upsampled to input size. Differentiable -> usable inside the CFA loss."""
        out = self.forward(x, return_features=True)
        feat = out["mal_features"]                                       # B,Cf,h,w (input/4)
        logit = out["malignancy"].sum()
        grads = torch.autograd.grad(logit, feat, create_graph=True)[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)                   # GAP weights
        cam = F.relu((weights * feat).sum(dim=1, keepdim=True))          # B,1,h,w
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        b = cam.shape[0]
        flat = cam.view(b, -1)
        mn = flat.min(1, keepdim=True)[0]; mx = flat.max(1, keepdim=True)[0]
        cam = (flat - mn) / (mx - mn + 1e-6)
        return cam.view(b, 1, *x.shape[-2:])
