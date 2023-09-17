import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

# get data
iris = sns.load_dataset("iris")

# organize data
data = torch.tensor(iris[iris.columns[0:4]].values, dtype=torch.float32)
labels = torch.zeros(len(data), dtype=torch.long)
# labels[iris.species == "setosa"] = 0
labels[iris.species == "versicolor"] = 1
labels[iris.species == "virginica"] = 2
# print(labels)

# making train and test sets using sklearn
train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.2)

# in this case avg of all set will be 1,
print()
print("avg of full data: ", torch.mean(labels.float()))
print("avg of train data: ", torch.mean(train_y.float()))
print("avg of test data: ", torch.mean(test_y.float()))
print()
print(f"full data: {data.shape}")
print(f"train data: {train_x.shape}")
print(f"test data: {test_x.shape}")
print()

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
epochs = 10000
for epoch in range(epochs):
    # forward pass
    output = ANNiris(train_x)

    # compute loss
    loss = lossFun(output, train_y)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# train accuracy
pred = ANNiris(train_x)
predLabel = torch.argmax(pred, axis=1)
train_accuracy = torch.mean((predLabel == train_y).float()) * 100

# test accuracy
pred = ANNiris(test_x)
predLabel = torch.argmax(pred, axis=1)
test_accuracy = torch.mean((predLabel == test_y).float()) * 100

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")