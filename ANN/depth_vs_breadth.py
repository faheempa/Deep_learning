import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# get data
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue="species")
# plt.show()

# organize data
data = torch.tensor(iris[iris.columns[0:4]].values, dtype=torch.float32)
labels = torch.zeros(len(data), dtype=torch.long)
# labels[iris.species == "setosa"] = 0
labels[iris.species == "versicolor"] = 1
labels[iris.species == "virginica"] = 2
# print(labels)

# ANN class
class ANNIris(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, nLayers):
        super().__init__()

        # dictionary of layers
        self.layers = nn.ModuleDict()
        self.nlayers = nLayers

        # input layer
        self.layers["input"] = nn.Linear(inputSize, hiddenSize)

        # hidden layers
        for i in range(nLayers):
            self.layers[f"hidden {i}"] = nn.Linear(hiddenSize, hiddenSize)

        # output layer
        self.layers["output"] = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        # input layer
        x = self.layers["input"](x)
        x = F.relu(x)

        # hidden layers
        for i in range(self.nlayers):
            x = self.layers[f"hidden {i}"](x)
            x = F.relu(x)

        # output layer
        x = self.layers["output"](x)
        x = F.softmax(x, dim=1)

        return x

# test the ANN
# ANN = ANNIris(4, 64, 3, 2)
# pred = ANN(data)
# print(pred.shape)
# print(ANN)
# print("ANN tested")

# function to train the modal
def trainModel(model):
    lossfun = nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # training
    epochs = 1000
    for i in range(epochs):
        # forward pass
        output = model(data)

        # compute loss
        loss = lossfun(output, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute accuracy
        matches = (torch.argmax(output, axis=1) == labels).type(torch.float32)
        accuracy = torch.mean(matches) * 100

    # final output
    pred = model(data)
    predLabel = torch.argmax(pred, axis=1)
    accuracy = torch.mean((predLabel == labels).float()) * 100

    # no of trainable parameters
    nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return nParams, accuracy

# test the model
# nParams, accuracy = trainModel(ANN)
# print(f"no of trainable parameters : {nParams} and accuracy : {accuracy}")

if __name__ == "__main__":
    # actual test
    numLayer = np.arange(1, 5)
    numUnits = np.arange(4, 101, 4)
    epochs = 1000

    # initialize output matrix
    nParams = np.zeros((len(numUnits), len(numLayer)))
    accuracies = np.zeros((len(numUnits), len(numLayer)))

    # test the model
    for i, unit in enumerate(numUnits):
        for j, layer in enumerate(numLayer):
            ANN = ANNIris(inputSize=4, hiddenSize=unit, outputSize=3, nLayers=layer)
            nParams[i, j], accuracies[i, j] = trainModel(ANN)

            print(f"unit: {unit}  units: {layer} tested")

    # plotting
    plt.plot(numUnits, accuracies, "o-")
    plt.xlabel("no of units")
    plt.ylabel("accuracy")
    plt.legend(numLayer)
    plt.show()

    plt.subplot(2, 2, 1)
    plt.plot(numUnits, np.mean(accuracies, axis=1), "o-", markerfacecolor="w")
    plt.title("Accuracy vs. Number of Units")
    plt.xlabel("Number of Units")
    plt.ylabel("Accuracy")
    
    plt.subplot(2, 2, 2)
    plt.plot(numLayer, np.mean(accuracies, axis=0), "o-", markerfacecolor="w")
    plt.title("Accuracy vs. Number of Layers")
    plt.xlabel("Number of Layers")
    plt.ylabel("Accuracy")
    plt.show()
    
    x = nParams.flatten()
    y = accuracies.flatten()
    # correlation btn them
    corr = np.corrcoef(x, y)[0, 1]
    plt.plot(x, y, "o")
    plt.xlabel("Number of Parameters")
    plt.ylabel("Accuracy")
    plt.title(f"Correlation: {corr:.3f}")
    plt.show()

   
