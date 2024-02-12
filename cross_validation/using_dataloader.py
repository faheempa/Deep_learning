import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# import data
iris = sns.load_dataset("iris")

# organize data
data = torch.tensor(iris[iris.columns[0:4]].values, dtype=torch.float32)
labels = torch.zeros(len(data), dtype=torch.long)
# labels[iris.species == "setosa"] = 0
labels[iris.species == "versicolor"] = 1
labels[iris.species == "virginica"] = 2
# print(labels)


# making train and test sets using sklearn
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2
)
# convert them into pytorch datasets
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)
# make dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
# checking shapes
for data, labels in train_loader:
    print(f"train data: {data.shape} train labels: {labels.shape}")
for data, labels in test_loader:
    print(f"test data: {data.shape} test labels: {labels.shape}")

# ANN and parameters
ANNiris = nn.Sequential(
    nn.Linear(4, 64),  # input layer
    nn.ReLU(),  # activation function
    nn.Linear(64, 64),  # hidden layer
    nn.ReLU(),  # activation function
    nn.Linear(64, 3),  # output layer
)
optimizer = torch.optim.SGD(ANNiris.parameters(), lr=0.01)  # optimizer
lossFun = nn.CrossEntropyLoss()  # loss function includes softmax as well

# training
epochs = 500
train_acc = []
test_acc = []

for epoch in range(epochs):
    # train
    for data, labels in train_loader:
        # forward pass
        output = ANNiris(data)

        # compute loss
        loss = lossFun(output, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # train accuracy
    pred = ANNiris(train_data)
    predLabel = torch.argmax(pred, axis=1)
    train_acc.append(torch.mean((predLabel == train_labels).float()) * 100)

    # test accuracy
    pred = ANNiris(test_data)
    predLabel = torch.argmax(pred, axis=1)
    test_acc.append(torch.mean((predLabel == test_labels).float()) * 100)

# plot
plt.figure(figsize=(10, 7))
plt.plot(train_acc, label="train accuracy")
plt.plot(test_acc, label="test accuracy")
plt.legend()
plt.show()

# final accuracy
pred = ANNiris(test_data)
predLabel = torch.argmax(pred, axis=1)
print(f"final accuracy: {torch.mean((predLabel == test_labels).float())*100}")
