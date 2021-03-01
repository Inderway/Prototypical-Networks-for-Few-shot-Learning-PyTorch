import numpy as np
s=slice(1,2)
a=[1,2,3,4,5,6]
print(a[s])

import torch

x = torch.tensor([[[1, 2, 3],[4,5,6]]])
print(x.size())
x=x.squeeze(0)
print(x)
print(x.size())