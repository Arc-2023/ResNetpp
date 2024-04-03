from torch import nn
import torch

cross = nn.CrossEntropyLoss()

input = torch.as_tensor([[0.5, 0.1], [0.5, 0.1]])
target = torch.as_tensor([1, 1])

output = cross(input, target)
print(output)
print(torch.as_tensor(0.6).long())
