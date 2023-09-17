# sampling
import numpy as np
import matplotlib.pyplot as plt
x=[1,2,4,2,6,3,2,8,3,6,3,6,12,11,2,4,7,2,4,6,7,8,12,10,3,11,4,14,6,2,4,6,7]
mean=np.mean(x)
print(mean)

means=[]
for i in range(1000):
    means.append(np.mean(np.random.choice(x, size=25, replace=True)))

plt.hist(means, bins=50)
plt.plot([mean,mean], [0,60])
plt.show()