# chances of getting wrong local minimum

import numpy as np
import matplotlib.pyplot as plt

def fx(x):
    return np.cos(2 * np.pi * x) + x**2

def deriv(x):
    return -2 * np.pi * np.sin(2 * np.pi * x) + 2 * x


x = np.linspace(-2, 2, 1000)
# Gradient descent algorithm
LR = 0.01
training_epochs = 100

for i in range(5):
    model_params = np.zeros((training_epochs, 2))
    local_min = np.random.choice(x, 1)[0]

    for i in range(training_epochs):
        gradient = deriv(local_min)
        local_min -= LR * gradient
        model_params[i, :] = local_min, gradient

    # Plot the function and gradient
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x, fx(x), label="f(x)")
    plt.plot(x, deriv(x), label="f'(x)")
    plt.scatter(local_min, fx(local_min), color="red", marker="o", label="Local Minimum")
    plt.xlabel("x")
    plt.ylabel("y / f'(x)")
    plt.title("Function and Gradient")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(model_params[:, 0], label="Local Minimum")
    plt.plot(model_params[:, 1], label="Gradient")
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.title("Parameter Updates")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
