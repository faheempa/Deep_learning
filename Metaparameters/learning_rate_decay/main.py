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
def createTheQwertyNet(initailLR, numepochs):
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
    optimizer = torch.optim.Adam(net.parameters(), lr=initailLR)
    stepSize = len(train_loader)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=stepSize, gamma=0.995
    )
    return net, lossfun, optimizer, scheduler


# a function that trains the model
def function2trainTheModel(lr, dynamicLR=False, numepochs=500):
    net, lossfun, optimizer, scheduler = createTheQwertyNet(lr, numepochs)
    losses = torch.zeros(numepochs)
    trainAcc = []
    testAcc = []
    lrs = []
    for epochi in range(numepochs):
        net.train()
        batchAcc = []
        batchLoss = []
        lrs.append(optimizer.param_groups[0]["lr"])
        for X, y in train_loader:
            yHat = net(X)
            loss = lossfun(yHat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update the learning rate
            if dynamicLR:
                scheduler.step()
            # save the loss
            batchLoss.append(loss.item())
            accuracyPct = 100 * torch.mean((torch.argmax(yHat, axis=1) == y).float())
            batchAcc.append(accuracyPct)
        trainAcc.append(np.mean(batchAcc))
        losses[epochi] = np.mean(batchLoss)
        net.eval()
        X, y = next(iter(test_loader))  # extract X,y from test dataloader
        with torch.no_grad():  # deactivates autograd
            yHat = net(X)
        testAcc.append(100 * torch.mean((torch.argmax(yHat, axis=1) == y).float()))
        print("epoch %d, testAcc=%f" % (epochi, testAcc[-1]))
    print()
    return trainAcc, testAcc, losses, lrs, net


# test scheduler
train_acc1, test_acc1, losses1, lrs1, net1 = function2trainTheModel(0.1, dynamicLR=True)
train_acc2, test_acc2, losses2, lrs2, net2 = function2trainTheModel(
    0.1, dynamicLR=False
)

# learning curves
plt.subplot(221)
plt.plot(lrs1, label="constant LR")
plt.plot(lrs2, label="dynamic LR")
plt.title("Learning rate decay")
plt.xlabel("Iteration")
plt.ylabel("Learning rate")

# plot losses
plt.subplot(222)
plt.plot(losses1, label="constant LR")
plt.plot(losses2, label="dynamic LR")
plt.title("Losses")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()

# plot accuracies
plt.subplot(223)
plt.plot(train_acc1, label="constant LR")
plt.plot(train_acc2, label="dynamic LR")
plt.title("Train accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend()

# plot accuracies
plt.subplot(224)
plt.plot(test_acc1, label="constant LR")
plt.plot(test_acc2, label="dynamic LR")
plt.title("Test accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
