import torch
from torch import nn

m = nn.ZeroPad2d(1)
input = torch.randn(2, 1, 3, 3)
output = m(input)
print('ok')
