# import
import torch
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 6)

# function returns the activation function
def getActivatonFun(act_type, x):
    act_fun = getattr(torch, act_type)
    return act_fun(x)


# activation functions
act_types = ["relu", "sigmoid", "tanh"]
x = torch.linspace(-3, 3, 100)

for act_type in act_types:
    y = getActivatonFun(act_type, x)
    plt.plot(x, y, label=act_type)
    
plt.title("Activation Functions")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# using torch.nn
def getActivatonFun(act_type):
    act_fun = getattr(torch.nn, act_type)
    return act_fun()

# activation functions
act_types = ["ReLU6", "Hardshrink", "LeakyReLU"]
x = torch.linspace(-3, 3, 100)

for act_type in act_types:
    act = getActivatonFun(act_type)
    y = act(x)
    plt.plot(x, y, label=act_type)

plt.title("Activation Functions")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# relu6
act = getActivatonFun("ReLU6")
x = torch.linspace(-10, 10, 100)
y = act(x)
plt.plot(x, y)
plt.title("ReLU6")
plt.xlabel("x")
plt.ylabel("y")
# mark the point where the slope changes
plt.scatter(6, 6, color="red")
plt.scatter(-6, 0, color="red")
plt.axvline(x=6, color="red", linestyle="--")
plt.axhline(y=6, color="red", linestyle="--")
plt.axvline(x=-6, color="red", linestyle="--")
plt.axhline(y=0, color="red", linestyle="--")
plt.show()

# leakyrelu
act = getActivatonFun("LeakyReLU")
x = torch.linspace(-1000, 100, 100)
y = act(x)
plt.plot(x, y)
plt.title("LeakyReLU")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

