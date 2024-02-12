import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# define the function
def peaks(x, y):
    x, y = np.meshgrid(x, y)
    return (
        3 * (1 - x) ** 2 * np.exp(-(x**2) - (y + 1) ** 2)
        - 10 * (x / 5 - x**3 - y**5) * np.exp(-(x**2) - y**2)
        - 1 / 3 * np.exp(-((x + 1) ** 2) - y**2)
    )


x, y = np.linspace(-3, 3, 201), np.linspace(-3, 3, 201)
Z = peaks(x, y)


# define the gradient
sx, sy = sp.symbols("sx, sy")
sZ = (
    3 * (1 - sx) ** 2 * sp.exp(-(sx**2) - (sy + 1) ** 2)
    - 10 * (sx / 5 - sx**3 - sy**5) * sp.exp(-(sx**2) - sy**2)
    - 1 / 3 * sp.exp(-((sx + 1) ** 2) - sy**2)
)
df_x = sp.lambdify((sx, sy), sp.diff(sZ, sx), "sympy")
df_y = sp.lambdify((sx, sy), sp.diff(sZ, sy), "sympy")
print(df_x(1, 1).evalf())
print(df_y(1, 1).evalf())

# random start point
localmin = np.random.rand(2) * 4 - 2
startPoint = localmin[:]
print("start point: ", startPoint)

# learning rate and iteration
lr = 0.01
epoch = 1000

# run through training
trajectory = np.zeros((epoch, 2))
for i in range(epoch):
    grad = np.array(
        [
            df_x(localmin[0], localmin[1]).evalf(),
            df_y(localmin[0], localmin[1]).evalf(),
        ],
        dtype=np.float64,
    )
    localmin -= lr * grad
    trajectory[i] = localmin.copy()

print("end point: ", localmin)

# plot the trajectory
plt.plot(trajectory[:, 0], trajectory[:, 1], "r.-")
plt.plot(trajectory[:, 0], trajectory[:, 1], "r.", markersize=3)
plt.plot(startPoint[0], startPoint[1], "bx", markersize=10)
plt.plot(localmin[0], localmin[1], "go", markersize=10)
plt.imshow(Z, extent=[-3, 3, -3, 3], vmin=-5, vmax=5, origin="lower")
plt.colorbar()
plt.show()
