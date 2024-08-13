import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from datasets_gen import CustomDataset
from sklearn.metrics import precision_score, recall_score, f1_score

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Remove the view line if the input is already the correct shape
        # x = x.view(-1, self.fc1.in_features) 
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
def train(net, trainloader, optimizer, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        print(f"Epoch {_+1}")
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
    all_labels = []
    all_predictions = []
    
    net.eval()
    with torch.no_grad():
        for feature, labels in testloader:
            # Should be (batch_size, input_size)
            outputs = net(feature)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = correct / len(testloader.dataset)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    return [loss, accuracy, precision, recall, f1]


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

    # Should be (batch_size, input_size)

    metrics = test(trained_model, test_loader)
    print(metrics)