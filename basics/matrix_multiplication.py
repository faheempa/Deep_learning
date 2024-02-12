import torch as tor

# matrix multiplication
m1 = tor.randn(3, 4)
m2 = tor.randn(4, 5)

print(tor.round(m1 @ m2))
