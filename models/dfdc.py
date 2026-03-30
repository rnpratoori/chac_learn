"""Neural network model for learning the free energy derivative df/dc."""

import torch
import torch.nn as nn


class FEDerivative(nn.Module):
    """Neural network to approximate the free energy derivatives df/dc and df/deta."""
    
    def __init__(self, hidden_size=200):
        super(FEDerivative, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        output = self.mlp(x)
        # Enforce zero mean ONLY on df/dc (index 0)
        # c is conserved, so df/dc is defined up to a constant.
        # eta is not conserved, so its mean value matters.
        dfdc = output[:, 0:1]
        dfdeta = output[:, 1:2]
        dfdc = dfdc - torch.mean(dfdc, dim=0, keepdim=True)
        return torch.cat([dfdc, dfdeta], dim=1)
