import numpy as np

# mean and varience
x=[1,2,3,4,5]
n=len(x)

print(np.mean(x))
print(sum(x)/n)
mean=np.mean(x)
print()

varience1 = (1/(n-1))*np.sum((x-mean)**2)
varience2 = np.var(x)
varience3 = np.var(x, ddof=1)
print(varience1)
print(varience2)
print(varience3)
print()

# as no of data increases np.var(x) is aproximatly equal to np.var(x, ddof=1)
N=10000
X=np.random.randint(low=0,high=20, size=N)
print(np.var(X))
print(np.var(X, ddof=1))