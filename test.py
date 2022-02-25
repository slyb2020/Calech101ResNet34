import torch

a = torch.randn(size=(1,3,4))
print(a.shape)
b = torch.randn(size=(1,3,4))
print(b.shape)
c = torch.concat([a,b], dim=0)
print(c.shape)