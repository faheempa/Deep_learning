import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap


def fx(x):
    return 3 * x**2 - 3 * x + 4


def deriv(x):
    return 6 * x - 3

epoch_lower = 10
epoch_high = 500
epoch_step = 300
lr_lower = 0.0001
lr_high = 0.01
lr_step = 300

x = np.linspace(-2, 2, 1000)
LRS = np.linspace(lr_lower, lr_high, epoch_step)
epochs = np.round(np.linspace(epoch_lower, epoch_high, lr_step))
res = [[0 for _ in range(epoch_step)] for _ in range(lr_step)]

for i, LR in enumerate(LRS):
    for j, training_epochs in enumerate(epochs):
        local_min = 0
        for _ in range(int(training_epochs)):
            gradient = deriv(local_min)
            local_min -= LR * gradient
        
        print(f"lr: {LR}, epochs: {training_epochs}, local_min: {local_min}")
        res[i][j] = np.round(local_min,2)

plt.plot(x, fx(x), label="f(x)")
plt.plot(x, deriv(x), label="f'(x)")
plt.legend()
plt.show()

# plot the res matrix
cmap=LinearSegmentedColormap.from_list('rg',["red",'orange','yellow', "green"])
plt.imshow(res, extent=[epoch_lower, epoch_high, lr_lower, lr_high], aspect='auto', origin="lower", cmap=cmap)
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.show()
