# softmax function
import torch.nn as nn
import torch as tor

z=[1,2,3,4,5]
softmax = nn.Softmax(dim=0)

sigmaT = softmax(tor.tensor(z, dtype=tor.float))
print(sigmaT)
print(sum(sigmaT))