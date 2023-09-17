# entropy
import numpy as np

# 25% probability of an event happening and 75% not happening
x = [0.25, 0.75]

H = 0
for p in x:
    H = H + (-p * np.log(p))

print("Entropy: ", H)

# explicitily wriiten as
p=0.25
H = -(p * np.log(p) + (1-p) * np.log(1-p))
print("Entropy: ", H)


# cross entropy
# probability of pic being a cat or not
p = [1, 0]
# probability of model making this prediction
q = [0.25, 0.75]

# cross entropy
H = 0
for i in range(len(p)):
    H -= p[i] * np.log(q[i])
print(H)

# explicitly written
H = -(p[0] * np.log(q[0]) + p[1] * np.log(q[1]))
print(H)

# p[1] is always 0, p[0] is always 1
H = -np.log(q[0])
print(H)