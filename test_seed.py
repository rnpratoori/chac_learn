import torch
from models.dfdc import FEDerivative
torch.manual_seed(12)
m1 = FEDerivative()
print(m1.state_dict()['model.0.weight'][0,0])

m2 = FEDerivative()
print(m2.state_dict()['model.0.weight'][0,0])
