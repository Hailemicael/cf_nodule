"""
Attribution baselines for the BIBM comparison table. Grad-CAM lives in the model
(differentiable, used by CFA). Here we wrap additional methods so all are scored
under the SAME faithfulness-first protocol in evaluate.py.

Requires `captum`. Each function returns a 1x1xPxP map in [0,1].
"""
import torch
import torch.nn.functional as F


def _norm(cam):
    b = cam.shape[0]; flat = cam.view(b, -1)
    mn = flat.min(1, keepdim=True)[0]; mx = flat.max(1, keepdim=True)[0]
    return ((flat - mn) / (mx - mn + 1e-6)).view_as(cam)


def _mal_forward(model):
    def f(x):
        return model(x)["malignancy"].unsqueeze(1)   # captum wants (B, n_out)
    return f


def integrated_gradients(model, x, steps=32):
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(_mal_forward(model))
    att = ig.attribute(x, baselines=torch.zeros_like(x), target=0, n_steps=steps)
    return _norm(att.abs())


def occlusion(model, x, window=8, stride=4):
    from captum.attr import Occlusion
    oc = Occlusion(_mal_forward(model))
    att = oc.attribute(x, target=0, sliding_window_shapes=(1, window, window),
                       strides=(1, stride, stride))
    return _norm(att.abs())


def gradcam(model, x):
    return model.gradcam_malignancy(x)


REGISTRY = {"gradcam": gradcam, "integrated_gradients": integrated_gradients,
            "occlusion": occlusion}
