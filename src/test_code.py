import numpy as np
import os
import torch
shape=2,2,3
x=np.arange(12).reshape(3,4)
print(x)
x=torch.from_numpy(x)
x=x.transpose(0,1).contiguous().view(shape)
print(x)