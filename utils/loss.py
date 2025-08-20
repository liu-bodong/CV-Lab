# Module for loss functions

import torch
import torch.nn as nn

__all__ = ['MSELoss']

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, inputs, targets):
        return nn.functional.mse_loss(inputs, targets)