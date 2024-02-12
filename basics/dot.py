import torch as tor
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 7, 2, 4, 8])

print(np.dot(a, b))

a = tor.tensor([1, 2, 3, 4, 5])
b = tor.tensor([5, 7, 2, 4, 8])

print(tor.dot(a, b))
