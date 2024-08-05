import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from datasets_gen import CustomDataset

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
def train(net, trainloader, optimizer, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for feature, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(feature), labels)
            loss.backward()
            optimizer.step()
    return net


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for feature, labels in testloader:
            outputs = net(feature)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def run_model(epochs: int, lr: float, model: SimpleNN, weights_list: list, train_dataset: CustomDataset, test_dataset: CustomDataset, momentum: float = 0.9):
    """A minimal (but complete) training loop"""

    # instantiate the model
    # model = SimpleNN(input_size, hidden_size, num_classes)

    # define optimiser with hyperparameters supplied
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # train for the specified number of epochs
    trained_model = train(model, train_loader, optim, epochs)
    
    weights_list.append(trained_model.state_dict())
    # training is completed, then evaluate model on the test set
    loss, accuracy = test(trained_model, test_loader)
    print(f"{loss = }")
    print(f"{accuracy = }")