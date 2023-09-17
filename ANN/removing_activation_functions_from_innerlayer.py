import torch as tor
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def binaryClassification(x, y, numepochs, LR, nperClust):
    # build model
    ANNclassify = nn.Sequential(
        nn.Linear(2, 1),  # input layer
        nn.Linear(1, 1),  # output layer
        nn.Sigmoid(),  # activation function
    )
    lossfun = nn.BCELoss()  # binary cross entropy loss
    optimizer = tor.optim.SGD(ANNclassify.parameters(), lr=LR)

    # training
    for _ in range(numepochs):
        # forward pass
        ycap = ANNclassify(x)

        # compute loss
        loss = lossfun(ycap, y)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # manually compute losses
    predictions = ANNclassify(x)
    predLabels = predictions > 0.5
    misClassified = np.where(predLabels != y)[0]
    accuracy = 100 * (1 - len(misClassified) / (2 * nperClust))

    return (predictions, predLabels, misClassified, accuracy)


if __name__ == "__main__":
    LRS = np.linspace(0.001, 0.1, 20)
    accuracies = np.zeros(len(LRS))
    for i, lr in enumerate(LRS):
        # create data
        nperClust = 100
        blur = 1
        A = [4, 2]
        B = [1, 1]
        clsA = [
            A[0] + np.random.randn(nperClust) * blur,
            A[1] + np.random.randn(nperClust) * blur,
        ]
        clsB = [
            B[0] + np.random.randn(nperClust) * blur,
            B[1] + np.random.randn(nperClust) * blur,
        ]
        label_np = np.vstack((np.zeros((nperClust, 1)), np.ones((nperClust, 1))))
        data_np = np.hstack((clsA, clsB)).T
        data = tor.tensor(data_np, dtype=tor.float)
        label = tor.tensor(label_np, dtype=tor.float)

        numepochs = 10000
        pred, predLabel, misclassified, accuracy = binaryClassification(
            data, label, numepochs, lr, nperClust
        )
        accuracies[i] = accuracy
        print(f"Learning rate = {lr:.4f}, accuracy = {accuracy:.2f}%")

    plt.plot(LRS, accuracies)
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.show()
