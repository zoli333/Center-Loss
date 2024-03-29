import torch
import torch.nn as nn
import torch.nn.functional as F

# create dummy input data
# input [100, 2]
# centers [10, 2]
# labels [100, 1]
input = torch.randn((100, 2))
centers = torch.randn((10, 2))
labels = torch.argmax(F.softmax(torch.randn((100, 10)), dim=1), dim=1, keepdim=True)
# print(labels.shape)

# test for one example
# i = 0
x = input[0]
test_label = labels[0]
c = x - centers[test_label]

test_label = 1
print(centers[test_label])
coords = torch.gather(centers, dim=0, index=torch.tensor([[test_label]]))
# just returns the x coordinate
print(coords)

# to get both x, y coordinates needs to expand the labels tensor (dim=1)
coords = torch.gather(centers, dim=0, index=torch.tensor([[test_label, test_label]]))
print(coords)

# test for all examples
labels = labels.expand(100, 2) # repeat the dim=1 in labels tensor
coords = torch.gather(centers, dim=0, index=labels)
print("input shape", input.shape)
print("coords shape: ", coords.shape)
c = input - coords

# loss function test
dist = c.pow(2).sum()
print(dist / c.size(0))

