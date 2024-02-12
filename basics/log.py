# log
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.0001, 1, 200)
# print(np.round(x,3))
logx = np.log(x)
# print(logx)
plt.plot(x, logx)
plt.show()

# log being the inverse of exponent

x = np.linspace(-10, 10, 50)
expx = np.exp(x)
logOfExpOfx = np.log(expx)

plt.scatter(logOfExpOfx, x)
plt.show()
