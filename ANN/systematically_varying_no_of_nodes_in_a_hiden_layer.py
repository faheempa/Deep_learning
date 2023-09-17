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

# organize data
label_np = np.hstack((np.zeros(nperClust), np.ones(nperClust), np.full(nperClust,2)))
data_np = np.hstack((clsA, clsB, clsC)).T
data = tor.tensor(data_np, dtype=tor.float)
labels = tor.tensor(label_np, dtype=tor.long)
# print(labels)

noOfNodes = 200
epochs = 100
accuracies = []
for i in range(1, noOfNodes):
    # ANN
    ANNiris = nn.Sequential(
        nn.Linear(2, i),  # input layer
        nn.ReLU(),  # activation function
        nn.Linear(i, i),  # hidden layer
        nn.ReLU(),  # activation function
        nn.Linear(i, 3),  # output layer
    )
    lossFun = nn.CrossEntropyLoss()  # loss function includes softmax as well
    optimizer = tor.optim.SGD(ANNiris.parameters(), lr=0.01)  # optimizer

    # training
    for epoch in range(epochs):
        # forward pass
        output = ANNiris(data)

        # compute loss
        loss = lossFun(output, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # final output
    pred = ANNiris(data)
    predLabel = tor.argmax(pred, axis=1)
    accuracy = tor.mean((predLabel == labels).float()) * 100
    accuracies.append(accuracy)
    print("Done with %d nodes in hidden layer and accuracy is %.2f" % (i, accuracy))

plt.plot(range(1, noOfNodes), accuracies)
plt.xlabel("Number of nodes in hidden layer")
plt.ylabel("Accuracy")
plt.show()

