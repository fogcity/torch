import torch
m = torch.nn.Linear(2, 1, False)
input = torch.randn(1, 2)
output = m(input)
print(output.size())
