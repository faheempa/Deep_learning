# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# import dataset (comes with colab!)
data = np.loadtxt(open("mnist_train_small.csv", "rb"), delimiter=",")

# extract labels (number IDs) and remove from data
labels = data[:, 0]
data = data[:, 1:]


def make_dataloaders(train_data_normalize, test_data_normalize):
    
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    if train_data_normalize:
        train_data = train_data / np.max(train_data)

    if test_data_normalize:
        test_data = test_data / np.max(test_data)

    # convert to tensor
    train_data = torch.tensor(train_data, dtype=torch.float)
    test_data = torch.tensor(test_data, dtype=torch.float)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)


    # convert to torch dataset and to dataloader
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    return train_loader, test_loader


# create a feedforward neural network
def create_model():
    class ANN(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(784, 128)
            self.hidden1 = nn.Linear(128, 64)
            self.hidden2 = nn.Linear(64, 32)
            self.output = nn.Linear(32, 10)
            self.dr = 0.5

        def forward(self, x):
            x = F.relu(self.input(x))
            x = F.dropout(x, p=self.dr, training=self.training)
            x = F.relu(self.hidden1(x))
            x = F.dropout(x, p=self.dr, training=self.training)
            x = F.relu(self.hidden2(x))
            x = F.dropout(x, p=self.dr, training=self.training)
            x = F.log_softmax(self.output(x), dim=1)
            return x

    model = ANN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    return model, optimizer, criterion


# # test the model
# # test_model, test_optimizer, test_criterion = create_model()
# # test_input = torch.randn(64, 784)
# # test_output = test_model(test_input)
# # print("Shape of input: %s" % str(test_input.shape))
# # print(torch.exp(test_output))  # output is log-probabilities, use exp to get probabilities


def train_model(train_loader, test_loader):
    print("--------------------------------------------------------")
    epochs = 50
    model, optimizer, criterion = create_model()
    losses = np.zeros(epochs)
    trainAcc = np.zeros(epochs)
    testAcc = np.zeros(epochs)

    for i in range(epochs):
        model.train()
        batchAcc = []
        bactchLoss = []
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batchAcc.append(
                100 * torch.mean((labels == torch.argmax(outputs, dim=1)).float())
            )
            bactchLoss.append(loss.item())
        losses[i] = np.mean(bactchLoss)
        trainAcc[i] = np.mean(batchAcc)

        # test model
        model.eval()
        x, y = next(iter(test_loader))
        with torch.no_grad():
            outputs = model(x)
            testAcc[i] = 100 * torch.mean((y == torch.argmax(outputs, dim=1)).float())
        print(f"Epoch: {i+1}/{epochs}, Train acc: {trainAcc[i]:.2f}")

    return model, losses, trainAcc, testAcc


# Experiment 1 
ep_name_1 = "Train & Test data are normalized"
train_loader, test_loader = make_dataloaders(train_data_normalize=True, test_data_normalize=True)
model_1, losses_1, trainAcc_1, testAcc_1 = train_model(train_loader, test_loader)

# Experiment 2 
ep_name_2 = "Only train data is normalized"
train_loader, test_loader = make_dataloaders(train_data_normalize=True, test_data_normalize=False)
model_2, losses_2, trainAcc_2, testAcc_2 = train_model(train_loader, test_loader)

# Experiment 3
ep_name_3 = "only test data is normalized"
train_loader, test_loader = make_dataloaders(train_data_normalize=False, test_data_normalize=True)
model_3, losses_3, trainAcc_3, testAcc_3 = train_model(train_loader, test_loader)

# Experiment 4 
ep_name_4 = "No normalization"
train_loader, test_loader = make_dataloaders(train_data_normalize=False, test_data_normalize=False)
model_4, losses_4, trainAcc_4, testAcc_4 = train_model(train_loader, test_loader)

# # plot
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.plot(trainAcc_1, label=ep_name_1)
plt.plot(trainAcc_2, label=ep_name_2)
plt.plot(trainAcc_3, label=ep_name_3)
plt.plot(trainAcc_4, label=ep_name_4)
plt.title("Training results")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Testing results")
plt.plot(testAcc_1, label=ep_name_1)
plt.plot(testAcc_2, label=ep_name_2)
plt.plot(testAcc_3, label=ep_name_3)
plt.plot(testAcc_4, label=ep_name_4)
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
