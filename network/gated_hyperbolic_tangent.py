import math
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GatedHyperbolicTangent(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w1, self.b1, self.w2, self.b2 = Parameter(torch.FloatTensor(out_features, in_features)), Parameter(
            torch.FloatTensor(out_features)), Parameter(torch.FloatTensor(out_features, in_features)), Parameter(
            torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.w1.data.uniform_(-stdv, stdv)
        self.b1.data.uniform_(-stdv, stdv)
        self.w2.data.uniform_(-2 * stdv, 1.5 * stdv)
        self.b2.data.uniform_(-2 * stdv, 1.5 * stdv)

    def forward(self, x):
        y = F.tanh(F.linear(x, self.w1, self.b1))
        g = F.sigmoid(F.linear(x, self.w2, self.b2))
        return y * g
