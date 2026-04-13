import torch
import numpy as np
import random
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(12)
from models.dfdc import FEDerivative
m = FEDerivative()
print(m.state_dict()['model.0.weight'][0,0])
