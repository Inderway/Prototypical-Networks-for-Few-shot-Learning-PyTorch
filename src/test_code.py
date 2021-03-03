import numpy as np
import os
import torch
from tqdm import tqdm

a=[1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3]
a=torch.tensor(a)
a_u=torch.unique(a)

print(a.eq(1).nonzero(as_tuple=False)[:5].squeeze(0))
