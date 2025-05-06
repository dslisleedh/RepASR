import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class NegativeCosineLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(NegativeCosineLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        pred = nn.functional.normalize(pred, dim=-1)
        target = nn.functional.normalize(target, dim=-1)

        loss = -torch.mean(torch.sum(pred * target, dim=-1))

        return self.loss_weight * loss
    