# min and max using numpy
import numpy as np
a=np.array([4,3,1,2,5])
print("min value: ",np.min(a))
print("min value index: ",np.argmin(a))
print("max value: ",np.max(a))
print("max value index: ",np.argmax(a))
print()

b=np.array([
    [4,1,3],
    [10,7,2], 
    [3,9,7]
])
print("min value: ",np.min(b))
print("min value index: ",np.argmin(b))
print("max value: ",np.max(b))
print("max value index: ",np.argmax(b))
print()
print("min value for each column: ",np.min(b,axis=0))
print("min value index for each column: ",np.argmin(b,axis=0))
print("max value for each column: ",np.max(b,axis=0))
print("max value index for each column: ",np.argmax(b,axis=0))
print()
print("min value for each row: ",np.min(b,axis=1))
print("min value index for each row: ",np.argmin(b,axis=1))
print("max value for each row: ",np.max(b,axis=1))
print("max value index for each row: ",np.argmax(b,axis=1))

# min and max using pytorch
import torch as tor

a=tor.tensor([4,3,1,2,5])
print("min value: ",tor.min(a))
print("min value index: ",tor.argmin(a))
print("max value: ",tor.max(a))
print("max value index: ",tor.argmax(a))
print()

b=tor.tensor([
    [4,1,2],
    [10,7,3], 
    [5,3,7]
])
print("min value: ",tor.min(b))
print("min value index: ",tor.argmin(b))
print("max value: ",tor.max(b))
print("max value index: ",tor.argmax(b))
print()
print("min value for each column: ",tor.min(b, axis=0).values)
print("min value index for each column: ",tor.min(b, axis=0).indices)
print("max value for each column: ",tor.max(b, axis=0).values)
print("max value index for each column: ",tor.max(b, axis=0).values)
print()
print("min value for each row: ",tor.min(b, axis=1).values)
print("min value index for each row: ",tor.min(b, axis=1).indices)
print("max value for each row: ",tor.max(b, axis=1).values)
print("max value index for each row: ",tor.max(b, axis=1).values)