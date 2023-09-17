# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt

# create data
nPerClust = 300
blur = 1

A = [1, 1]
B = [5, 1]
C = [4, 3]

a = [A[0] + np.random.randn(nPerClust) * blur, A[1] + np.random.randn(nPerClust) * blur]
b = [B[0] + np.random.randn(nPerClust) * blur, B[1] + np.random.randn(nPerClust) * blur]
c = [C[0] + np.random.randn(nPerClust) * blur, C[1] + np.random.randn(nPerClust) * blur]

# true labels
labels_np = np.hstack(
    (np.zeros((nPerClust)), np.ones((nPerClust)), 1 + np.ones((nPerClust)))
)

# concatanate into a matrix
data_np = np.hstack((a, b, c)).T

# convert to a pytorch tensor
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).long()  # note: "long" format for CCE

# show the data
fig = plt.figure(figsize=(5, 5))
plt.plot(
    data[np.where(labels == 0)[0], 0],
    data[np.where(labels == 0)[0], 1],
    "bs",
    alpha=0.5,
)
plt.plot(
    data[np.where(labels == 1)[0], 0],
    data[np.where(labels == 1)[0], 1],
    "ko",
    alpha=0.5,
)
plt.plot(
    data[np.where(labels == 2)[0], 0],
    data[np.where(labels == 2)[0], 1],
    "r^",
    alpha=0.5,
)
plt.title("The qwerties!")
plt.xlabel("qwerty dimension 1")
plt.ylabel("qwerty dimension 2")
plt.show()

# split the data
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.1
)
# convert to pytorch dataset object
train_data = torch.utils.data.TensorDataset(train_data, train_labels)
test_data = torch.utils.data.TensorDataset(test_data, test_labels)

# dataloader objects
batchsize = 32
train_loader = DataLoader(
    train_data, batch_size=batchsize, shuffle=True, drop_last=True
)
test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])


# create a class for the model
def createTheQwertyNet(L2lambda):
    class qwertyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(2, 8)
            self.fc1 = nn.Linear(8, 8)
            self.output = nn.Linear(8, 3)

        # forward pass
        def forward(self, x):
            x = F.relu(self.input(x))
            x = F.relu(self.fc1(x))
            return self.output(x)

    net = qwertyNet()
    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=L2lambda)
    return net, lossfun, optimizer


# a function that trains the model
def function2trainTheModel(L2lambda):
    net, lossfun, optimizer = createTheQwertyNet(L2lambda)
    losses = torch.zeros(numepochs)
    trainAcc = []
    testAcc = []
    for epochi in range(numepochs):
        net.train()
        batchAcc = []
        batchLoss = []
        for X, y in train_loader:
            yHat = net(X)
            loss = lossfun(yHat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batchLoss.append(loss.item())
            matches = torch.argmax(yHat, axis=1) == y  # booleans (false/true)
            matchesNumeric = matches.float()  # convert to numbers (0/1)
            accuracyPct = 100 * torch.mean(matchesNumeric)  # average and x100
            batchAcc.append(accuracyPct)  # add to list of accuracies
        trainAcc.append(np.mean(batchAcc))
        losses[epochi] = np.mean(batchLoss)
        net.eval()
        X, y = next(iter(test_loader))  # extract X,y from test dataloader
        with torch.no_grad():  # deactivates autograd
            yHat = net(X)
        testAcc.append(100 * torch.mean((torch.argmax(yHat, axis=1) == y).float()))
    return trainAcc, testAcc, losses, net


# experiment with different L2 lambdas
l2lambdas = np.linspace(0, 0.1, 6)
numepochs = 50
accuracyResultsTrain = np.zeros((numepochs, len(l2lambdas)))
accuracyResultsTest = np.zeros((numepochs, len(l2lambdas)))
for li in range(len(l2lambdas)):
    trainAcc, testAcc, losses, net = function2trainTheModel(l2lambdas[li])
    accuracyResultsTrain[:, li] = trainAcc
    accuracyResultsTest[:, li] = testAcc

# plo results
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(accuracyResultsTrain, linewidth=2)
ax[0].set_title("Train accuracy")
ax[1].plot(accuracyResultsTest, linewidth=2)
ax[1].set_title("Test accuracy")
leglabels = [np.round(i, 2) for i in l2lambdas]
for i in range(2):
    ax[i].legend(leglabels)
    ax[i].set_xlabel("Epoch")
    ax[i].set_ylabel("Accuracy (%)")
    ax[i].set_ylim([30, 101])
    ax[i].grid()
plt.show()
