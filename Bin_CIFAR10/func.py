import torch.nn as nn
import torch.nn.functional as F

# Ref: https://discuss.pytorch.org/t/calculating-the-entropy-loss/14510/2
class Entropy(nn.Module):
  def __init__(self):
      super(Entropy, self).__init__()
  def forward(self, x):
      b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
      b = -1.0 * b.sum()
      return b