import torch as tor
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, noHiddenLayers=1):
        super().__init__()
        # dictionary of layers
        self.layers = nn.ModuleDict()
        self.noHiddenLayers = noHiddenLayers
        # input layer
        self.layers["input"] = nn.Linear(inputSize, hiddenSize)
        # hidden layers
        for i in range(noHiddenLayers):
            self.layers[f"hidden {i}"] = nn.Linear(hiddenSize, hiddenSize)
        # output layer
        self.layers["output"] = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        # input layer
        x = self.layers["input"](x)
        x = F.relu(x)
        # hidden layers
        for i in range(self.noHiddenLayers):
            x = self.layers[f"hidden {i}"](x)
            x = F.relu(x)
        # output layer
        x = self.layers["output"](x)
        x = F.sigmoid(x)
        return x


def binaryClassification(x, y, numepochs, LR, nperClust):
    # build model
    # ANNclassify = nn.Sequential(
    #     nn.Linear(2, 16),  # input layer
    #     nn.ReLU(),  
    #     nn.Linear(16, 16), 
    #     nn.ReLU(),  
    #     nn.Linear(16, 16),  
    #     nn.ReLU(), 
    #     nn.Linear(16, 1),  # output layer
    #     nn.Sigmoid(),  # activation function
    # )
    ANNclassify = ANN(inputSize=2, hiddenSize=16, outputSize=1, noHiddenLayers=2)

    lossfun = nn.BCELoss()  # binary cross entropy loss
    optimizer = tor.optim.SGD(ANNclassify.parameters(), lr=LR)
    losses = tor.zeros(numepochs)

    # training
    for epoch in range(numepochs):
        # forward pass
        ycap = ANNclassify(x)

        # compute loss
        loss = lossfun(ycap, y)
        losses[epoch] = loss
        print(f"Epoch {epoch+1:03}/{numepochs}, loss = {loss.item():.4f}")

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot losses
    plt.plot(losses.detach(), "o", markerfacecolor="w", linewidth=0.1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Final loss = {loss.item():.4f}")
    plt.show()

    # manually compute losses
    predictions = ANNclassify(x)
    predLabels = predictions > 0.5
    misClassified = np.where(predLabels != y)[0]
    accuracy = 100 * (1 - len(misClassified) / (2 * nperClust))

    return (predictions, predLabels, misClassified, accuracy)


if __name__ == "__main__":
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
    lr = 0.01
    pred, predLabel, misclassified, accuracy = binaryClassification(
        data, label, numepochs, lr, nperClust
    )

    # visualize data
    plt.subplot(1, 2, 1)
    plt.plot(clsA[0], clsA[1], "o", label="class A")
    plt.plot(clsB[0], clsB[1], "o", label="class B")
    plt.legend()
    plt.title("Data")
    
    # pred data
    plt.subplot(1, 2, 2)
    plt.plot(
        data_np[predLabel[:, 0] == 0, 0],
        data_np[predLabel[:, 0] == 0, 1],
        "o",
        label="class A",
    )
    plt.plot(
        data_np[predLabel[:, 0] == 1, 0],
        data_np[predLabel[:, 0] == 1, 1],
        "o",
        label="class B",
    )
    plt.legend()
    plt.title("Predicted data")
    plt.show()

    print(f"Accuracy = {accuracy:.2f}%")
