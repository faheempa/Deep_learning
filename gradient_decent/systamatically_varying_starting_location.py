import numpy as np
import matplotlib.pyplot as plt


def fx(x):
    return np.sin(x) * np.exp(-(x**2) * 0.05)


def deriv(x):
    return np.cos(x) * np.exp(-(x**2) * 0.05) - 0.1 * x * np.sin(x) * np.exp(
        -(x**2) * 0.05
    )


x = np.linspace(-2 * np.pi, 2 * np.pi, 500)
start_point = np.linspace(-2 * np.pi, 2 * np.pi, 500)
minimas = []
for local_min in start_point:
    LR = 0.01
    training_epochs = 1000

    for i in range(training_epochs):
        gradient = deriv(local_min)
        local_min -= LR * gradient

    minimas.append(local_min)

plt.plot(x, fx(x), label="f(x)")
plt.plot(x, deriv(x), label="f'(x)")
plt.legend()
plt.show()
plt.plot(x, minimas)
plt.show()
