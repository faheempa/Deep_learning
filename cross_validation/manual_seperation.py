import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# get data
iris = sns.load_dataset("iris")
# sns.pairplot(iris, hue="species")
# plt.show()

# organize data
data = torch.tensor(iris[iris.columns[0:4]].values, dtype=torch.float32)
labels = torch.zeros(len(data), dtype=torch.long)
# labels[iris.species == "setosa"] = 0
labels[iris.species == "versicolor"] = 1
labels[iris.species == "virginica"] = 2
# print(labels)

# making train and test sets manually
propTraining = 0.8
nTraining = int(len(labels)*propTraining)
trainTestBool = np.zeros(len(labels), dtype=bool)
randomIndexOfTraining = np.random.choice(range(len(labels)), size=nTraining, replace=False)
trainTestBool[randomIndexOfTraining]=True

# in this case avg of all set will be 1,
print()
print("avg of full data: ", torch.mean(labels.float()))
print("avg of train data: ", torch.mean(labels[trainTestBool].float()))
print("avg of test data: ", torch.mean(labels[~trainTestBool].float()))
print()
print(f"full data: {data.shape}")
print(f"train data: {data[trainTestBool].shape}")
print(f"test data: {data[~trainTestBool].shape}")
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
    output = ANNiris(data[trainTestBool])

    # compute loss
    loss = lossFun(output, labels[trainTestBool])

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # compute accuracy
    matches = (torch.argmax(output, axis=1) == labels[trainTestBool]).type(torch.float32)

# train accuracy
pred = ANNiris(data[trainTestBool])
predLabel = torch.argmax(pred, axis=1)
train_accuracy = torch.mean((predLabel == labels[trainTestBool]).float()) * 100

# test accuracy
pred = ANNiris(data[~trainTestBool])
predLabel = torch.argmax(pred, axis=1)
test_accuracy = torch.mean((predLabel == labels[~trainTestBool]).float()) * 100


print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")