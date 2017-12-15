import torch
import torch.nn.functional as F
from torch.nn import Module, Linear


class GatedHyperbolicTangent(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1, self.linear2 = Linear(in_features, out_features), Linear(in_features, out_features)

    def forward(self, x):
        y = self.linear1(x)
        y = F.tanh(y)
        g = self.linear2(x)
        g = F.sigmoid(g)
        return torch.mul(y, g)
