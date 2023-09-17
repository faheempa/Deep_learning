import numpy as np
import matplotlib.pyplot as plt


def fx(x):
    return 3 * x**2 - 3 * x + 4


def deriv(x):
    return 6 * x - 3


x = np.linspace(-2, 2, 1000)

for _ in range(5):
    # Gradient descent algorithm
    LR = 0.01
    training_epochs = 100

    model_params = np.zeros((training_epochs, 2))
    local_min = np.random.choice(x, 1)[0]
    start_point = local_min.copy()

    for i in range(training_epochs):
        gradient = deriv(local_min)
        local_min -= LR * gradient
        model_params[i, :] = local_min, gradient

    # Plot the function and gradient
    plt.plot(x, fx(x))
    plt.plot(x, deriv(x))
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend(["y", "dy"])
    plt.scatter(local_min, fx(local_min))
    plt.scatter(start_point, fx(start_point))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(2):
        ax[i].plot(model_params[:, i])
        ax[i].set_xlabel("Iterations")

    ax[0].set_title(f"Final estimated minimum: {local_min}")
    ax[0].set_ylabel("Local minimum")
    ax[1].set_title(f"Derivate at final localmin: {gradient}")
    ax[1].set_ylabel("Derivative")
    plt.show()
