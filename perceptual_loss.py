# perceptual_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class VGGPerceptual(nn.Module):
    """
    Frozen VGG19 feature extractor for perceptual loss.
    Expects inputs in [0, 1] range, normalized roughly like ImageNet.
    """
    def __init__(self, layers=("relu2_2", "relu3_4")):
        super().__init__()

        weights = VGG19_Weights.IMAGENET1K_V1
        vgg = vgg19(weights=weights).features

        # Map friendly names to layer indices in VGG19 features
        name_to_idx = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_4": 17,
            "relu4_4": 26,
            "relu5_4": 35,
        }
        self.layers_idx = [name_to_idx[name] for name in layers]

        max_idx = max(self.layers_idx)
        self.backbone = vgg[: max_idx + 1]

        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        x: [B, 3, H, W], in [0, 1]
        returns list of feature maps at selected layers
        """
        feats = []
        out = x
        for i, layer in enumerate(self.backbone):
            out = layer(out)
            if i in self.layers_idx:
                feats.append(out)
        return feats


def perceptual_loss(vgg: VGGPerceptual,
                    recon: torch.Tensor,
                    target: torch.Tensor,
                    layer_weights=None) -> torch.Tensor:
    """
    vgg: frozen VGGPerceptual
    recon, target: [B, 3, H, W] in [0, 1]
    returns scalar loss tensor
    """
    if layer_weights is None:
        layer_weights = [1.0] * len(vgg.layers_idx)

    # Don't track gradients through target features
    with torch.no_grad():
        target_feats = vgg(target)

    recon_feats = vgg(recon)

    loss = 0.0
    for w, rf, tf in zip(layer_weights, recon_feats, target_feats):
        loss = loss + w * F.l1_loss(rf, tf)

    return loss