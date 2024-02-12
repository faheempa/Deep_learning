# iris dataset classification using ANN

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# get data
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue="species")
plt.show()
# print(iris.head())

# organize data
data = torch.tensor(iris[iris.columns[0:4]].values, dtype=torch.float32)
labels = torch.zeros(len(data), dtype=torch.long)
# labels[iris.species == "setosa"] = 0
labels[iris.species == "versicolor"] = 1
labels[iris.species == "virginica"] = 2
print(labels)

# ANN
ANNiris = nn.Sequential(
    nn.Linear(4, 64),  # input layer
    nn.ReLU(),  # activation function
    nn.Linear(64, 64),  # hidden layer
    nn.ReLU(),  # activation function
    nn.Linear(64, 3),  # output layer
)
lossFun = nn.CrossEntropyLoss()  # loss function includes softmax as well
optimizer = torch.optim.SGD(ANNiris.parameters(), lr=0.01)  # optimizer

# training
epochs = 1000
losses = torch.zeros(epochs)
accuracies = torch.zeros(epochs)
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
    matches = (torch.argmax(output, axis=1) == labels).type(torch.float32)
    accuracy = torch.mean(matches) * 100
    accuracies[epoch] = accuracy.detach()
    print("epoch: %d, loss: %.2f, accuracy: %.2f" % (epoch, loss, accuracy))

# final output
pred = ANNiris(data)
predLabel = torch.argmax(pred, axis=1)
accuracy = torch.mean((predLabel == labels).float()) * 100

# plot
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Final accuracy: %.2f" % accuracy)
plt.show()

# confirming that sum of each row went to softmax output is 1
sm = nn.Softmax(dim=1)
print(torch.sum(sm(pred), axis=1))

# plot output with and without softmax
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(pred.detach().numpy())
plt.legend(iris.species.unique())
plt.title("without softmax")

plt.subplot(1, 2, 2)
plt.plot(sm(pred).detach().numpy())
plt.legend(iris.species.unique())
plt.title("with softmax")
plt.show()