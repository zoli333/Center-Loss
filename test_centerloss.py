import torch
import torch.nn as nn
import torch.nn.functional as F

# create dummy input data
# input [100, 2]
# centers [10, 2]
# labels [100, 10]
input = torch.randn((100, 2))
centers = torch.randn((10, 2))
labels = torch.argmax(F.softmax(torch.randn((100, 10)), dim=1), dim=1, keepdim=True)
# print(labels.shape)

# test for one example
# i = 0
x = input[0]
label = labels[0]
c = x - centers[label]

# test for all examples
labels = labels.expand(100, 2)
coords = torch.gather(centers, dim=0, index=labels)
c = input - coords

# loss function test
dist = c.pow(2).sum()
print(dist / c.size(0))

