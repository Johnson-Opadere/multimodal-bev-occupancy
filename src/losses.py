"""
losses.py
----------
Defines losses for BEV occupancy prediction.
Includes:
- Binary Cross Entropy (BCE)
- Dice Loss
- Hybrid Loss (BCE + Dice)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------
# Dice Loss
# -----------------------------------------------------------
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# -----------------------------------------------------------
# Hybrid Loss (BCE + Dice)
# -----------------------------------------------------------
class HybridLoss(nn.Module):
    """Hybrid loss = BCEWithLogits + Dice."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target, smooth=1e-6):
        # Ensure shapes align
        if target.ndim == 3:
            target = target.unsqueeze(1)
        target = target.float()

        # --- Binary cross-entropy with logits (safe under autocast)
        bce = F.binary_cross_entropy_with_logits(pred, target)

        # --- Dice uses sigmoid manually
        pred_sig = torch.sigmoid(pred).clamp(1e-6, 1 - 1e-6)
        intersection = (pred_sig * target).sum(dim=(2, 3))
        dice = 1 - (2. * intersection + smooth) / (
            pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth
        )
        dice = dice.mean()

        # --- Weighted hybrid
        return self.bce_weight * bce + self.dice_weight * dice
