import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        bce = self.bce(preds, targets)
        preds = (preds > 0.5).float()

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

        return 1 - dice + bce

def deep_supervision_loss(outputs, target, loss_fn, weights=[0.1, 0.2, 0.3, 0.4]):
    assert len(outputs) == len(weights)
    loss = 0.0
    for out, w in zip(outputs, weights):
        loss += w * loss_fn(out, target)
    return loss
