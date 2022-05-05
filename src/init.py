import torch as torch
m = torch.nn.Softmax(dim=1)
input = torch.randn(2, 3)
print(input)
output = m(input)
print(output)
