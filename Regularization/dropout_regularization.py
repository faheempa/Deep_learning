# Dropout is a technique where randomly selected neurons are ignored during training. 
# They are “dropped out” randomly. This means that their contribution to the activation 
# of downstream neurons is temporally removed on the forward pass, and any weight updates 
# are not applied to the neuron on the backward pass

import torch
import torch.nn as nn
import torch.nn.functional as F

# define dropout instance
print("dropout instance in training mode")
dropout = nn.Dropout(p=0.5)
x = torch.ones(10)
y = dropout(x)
print(y)
print(torch.mean(y))
print()

# switch to evaluation mode
print("dropout instance in evaluation mode")
dropout.eval()
y = dropout(x)
print(y)
print(torch.mean(y))
print()

# using F.dropout
print("F.dropout in training mode")
x = torch.ones(10)
y = F.dropout(x, p=0.5, training=True)
print(y)
print(torch.mean(y))
print("F.dropout in evaluation mode")
y = F.dropout(x, p=0.5, training=False)
print(y)
print(torch.mean(y))
print()

# toggling between training and evaluation mode
print("toggling between training and evaluation mode")
print("dropout instance in training mode")
dropout.train()
y = dropout(x)
print(y)

print("dropout instance in evaluation mode(should be all ones)")
dropout.eval()
y = dropout(x)
print(y)

print("back to training mode")
dropout.train()
y = dropout(x)
print(y)