import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch


df = pd.read_csv("../../../datasets/Machine Predictive Maintenance Classification/binary_classification.csv", index_col=[0])


X = df.drop(columns='Target')
Y = df.drop(columns=["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"])


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# def get_device():
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         device_name = torch.cuda.get_device_name(0)
#         print(f"Using GPU: {device_name}")
#     else:
#         device = torch.device("cpu")
#         print("Using CPU")
#     return device

# device = get_device()


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.reset_index(drop=True)  # Reset indices to avoid indexing issues
        self.y = y.reset_index(drop=True)  # Reset indices to avoid indexing issues

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        try:
            X_tensor = torch.tensor(self.X.iloc[idx].values, dtype=torch.float32)
            y_tensor = torch.tensor(self.y.iloc[idx], dtype=torch.long)
            return X_tensor, y_tensor
        except TypeError:
            self._check_indexing_error(idx)
        except Exception as e:
            print(f"Unexpected error: {e}, Index: {idx}")

    def _check_indexing_error(self, idx):
        if isinstance(idx, (list, tuple, pd.Index)):
            raise IndexError("Invalid index provided. Index should be an integer.")
        raise


train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


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
    
input_size = X.shape[1]
hidden_size = 10
num_classes = 2

model = SimpleNN(input_size, hidden_size, num_classes)


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


def run_centralised(epochs: int, lr: float, momentum: float = 0.9):
    """A minimal (but complete) training loop"""

    # instantiate the model
    model = SimpleNN(input_size, hidden_size, num_classes)

    # define optimiser with hyperparameters supplied
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # train for the specified number of epochs
    trained_model = train(model, train_loader, optim, epochs)

    # training is completed, then evaluate model on the test set
    loss, accuracy = test(trained_model, test_loader)
    print(f"{loss = }")
    print(f"{accuracy = }")


run_centralised(epochs=5, lr=0.01)
