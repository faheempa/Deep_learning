import torch as tor
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

nperClust = 100
blur = 1.25
C = [5, 5]
A = [0, 5]
B = [2, 1]
clsA = [
    A[0] + np.random.randn(nperClust) * blur,
    A[1] + np.random.randn(nperClust) * blur,
]
clsB = [
    B[0] + np.random.randn(nperClust) * blur,
    B[1] + np.random.randn(nperClust) * blur,
]
clsC = [
    C[0] + np.random.randn(nperClust) * blur,
    C[1] + np.random.randn(nperClust) * blur,
]
label_np = np.vstack((np.zeros((nperClust, 1)), np.ones((nperClust, 1)), np.full((nperClust, 1),2)))
data_np = np.hstack((clsA, clsB, clsC)).T
data = tor.tensor(data_np, dtype=tor.float)
labels = tor.tensor(label_np, dtype=tor.float)

# visualize data
plt.subplot(2, 2, 1)
plt.plot(clsA[0], clsA[1], "o", label="class A")
plt.plot(clsB[0], clsB[1], "o", label="class B")
plt.plot(clsC[0], clsC[1], "o", label="class C")
plt.legend()
plt.title("Data")
# plt.show()

# organize data
label_np = np.hstack((np.zeros(nperClust), np.ones(nperClust), np.full(nperClust,2)))
data_np = np.hstack((clsA, clsB, clsC)).T
data = tor.tensor(data_np, dtype=tor.float)
labels = tor.tensor(label_np, dtype=tor.long)
# print(labels)

# ANN
ANNiris = nn.Sequential(
    nn.Linear(2, 64),  # input layer
    nn.ReLU(),  # activation function
    nn.Linear(64, 64),  # hidden layer
    nn.ReLU(),  # activation function
    nn.Linear(64, 3),  # output layer
)
lossFun = nn.CrossEntropyLoss()  # loss function includes softmax as well
optimizer = tor.optim.SGD(ANNiris.parameters(), lr=0.01)  # optimizer

# training
epochs = 5000
losses = tor.zeros(epochs)
accuracies = tor.zeros(epochs)
for epoch in range(epochs):
    # forward pass
    output = ANNiris(data)

    # compute loss
    loss = lossFun(output, labels)
    losses[epoch] = loss.detach()

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # compute accuracy
    matches = (tor.argmax(output, axis=1) == labels).type(tor.float32)
    accuracy = tor.mean(matches) * 100
    accuracies[epoch] = accuracy.detach()
    print("epoch: %d, loss: %.2f, accuracy: %.2f" % (epoch, loss, accuracy))

# final output
pred = ANNiris(data)
predLabel = tor.argmax(pred, axis=1)
accuracy = tor.mean((predLabel == labels).float()) * 100

# visualize Predicted data
plt.subplot(2, 2, 2)
plt.plot(data[predLabel==0, 0], data[predLabel==0, 1], 'o', label="class A")
plt.plot(data[predLabel==1, 0], data[predLabel==1, 1], 'o', label="class B")
plt.plot(data[predLabel==2, 0], data[predLabel==2, 1], 'o', label="class C")
plt.title("Predicted Data")
plt.legend()
# plt.show()

# ploting accuracy
plt.subplot(2, 2, 3)
plt.plot(accuracies)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Final accuracy: %.2f" % accuracy)
# plt.show()

# plot output 
plt.subplot(2, 2, 4)
sm = nn.Softmax(dim=1)
plt.plot(sm(pred).detach().numpy())
plt.show() 