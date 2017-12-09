import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module


class VqaLoss(Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        if self.weight is not None:
            return binary_cross_entropy_with_logits_minus(input, target, Variable(self.weight), self.size_average)
        else:
            return binary_cross_entropy_with_logits_minus(input, target, size_average=self.size_average)


def binary_cross_entropy_with_logits_minus(input, target, weight=None, size_average=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val - ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


# = -[z * log(y) - (1-z) * log(1-y)]
# = z * log(1 + exp(-x)) + (1-z) * [-x - log(1 + exp(-x))]
# = z * log(1 + exp(-x)) - x - log(1 + exp(-x)) + z_x + z * log(1 + exp(-x))
# = - x + z_x + log(1 + exp(-x)) + 2 * z * log(1 + exp(-x))