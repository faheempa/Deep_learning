# derivatives
import sympy
from sympy.abc import x

fx = 3 * x**2 + 6 * x + 10
dfx = sympy.diff(fx, x)
print(dfx)
