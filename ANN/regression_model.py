import torch as tor
import torch.nn as nn
import matplotlib.pyplot as plt

def regressionFunc(x,y,numepochs,LR):

    # build model
    regANN = nn.Sequential(
        nn.Linear(1, 1),  # input layer
        nn.ReLU(),  # activation function
        nn.Linear(1, 1),  # output layer
    )

    # learning rate, loss function, optimizer, numepochs and losses
    lossfun = nn.MSELoss()
    # optimizer: type of gradient descent we want to use
    optimizer = tor.optim.SGD(regANN.parameters(), lr=LR)
    losses = tor.zeros(numepochs)

    # training
    for epoch in range(numepochs):
        # forward pass
        ycap = regANN(x)

        # compute loss
        loss = lossfun(ycap, y)
        losses[epoch] = loss
        print(f"Epoch {epoch+1:03}/{numepochs}, loss = {loss.item():.4f}")

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # manually compute losses
    predictions = regANN(x)
    testloss = (predictions - y).pow(2).mean()

    return (predictions, losses, testloss)

if __name__ == "__main__":
    # create data
    N = 200
    x = tor.randn(N, 1)
    y = x + tor.randn(N, 1) / 2
    numepochs = 10000
    lr = 0.01

    predictions, losses, testloss = regressionFunc(x,y,numepochs,lr)
    plt.plot(losses.detach(), "o", markerfacecolor="w", linewidth=0.1)
    plt.plot(numepochs, testloss.detach(), "bo")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Final loss = {testloss.item()}")
    plt.show()
    # ploting the data
    plt.plot(x, y, "s", label="real data")
    plt.plot(x, predictions.detach(), "ro", label="predicted data")
    plt.legend()
    plt.show()