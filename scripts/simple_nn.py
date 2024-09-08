import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from datasets_gen import CustomDataset
from sklearn.metrics import precision_score, recall_score, f1_score

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
    
def train(net, trainloader, optimizer, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    for _ in range(epochs):
        for feature, labels in trainloader:
            feature, labels = feature.to(device), labels.to(device)  # Move data to GPU
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

    net.to(device)
    net.eval()
    with torch.no_grad():
        for feature, labels in testloader:
            # Move data to GPU if available
            feature, labels = feature.to(device), labels.to(device)
            outputs = net(feature)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())  # Move to CPU for evaluation
            all_predictions.extend(predicted.cpu().numpy())  # Move to CPU for evaluation
    
    accuracy = correct / len(testloader.dataset)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    return [loss, accuracy, precision, recall, f1]



def run_model(epochs: int, lr: float, model: SimpleNN, weights_list: list, train_dataset: CustomDataset, test_dataset: CustomDataset, momentum: float = 0.9):
    """A minimal (but complete) training loop"""

    # Move model to GPU
    model = model.to(device)

    # Define optimiser with hyperparameters supplied
    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Train for the specified number of epochs
    trained_model = train(model, train_loader, optim, epochs)
    
    weights_list.append(trained_model.state_dict())

    # After training, evaluate model on the test set
    # metrics = test(trained_model, test_loader)
    # print(metrics)