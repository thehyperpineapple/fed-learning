import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from torchvision.models import get_model_weights
import copy


df = pd.read_csv("../../../datasets/Machine Predictive Maintenance Classification/binary_classification.csv", index_col=[0])

X = df.drop(columns='Target')
Y = df['Target']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


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


# Split the train_dataset into 5 subsets
num_subsets = 5
subset_size = len(train_dataset) // num_subsets
indices = list(range(len(train_dataset)))
np.random.shuffle(indices)

subsets = []
for i in range(num_subsets):
    start_idx = i * subset_size
    end_idx = (i + 1) * subset_size if i != num_subsets - 1 else len(train_dataset)
    subset_indices = indices[start_idx:end_idx]
    subsets.append(Subset(train_dataset, subset_indices))


# Neural Network
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



def train(net, trainloader, optimizer, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    return net


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def run_model(epochs: int, lr: float, model: SimpleNN, weights_list: list, train_dataset: CustomDataset, test_dataset: CustomDataset, momentum: float = 0.9):

    # define optimiser with hyperparameters supplied
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # train for the specified number of epochs
    trained_model = train(model, train_loader, optim, epochs)

    # save model weights in a list
    weights_list.append(trained_model.state_dict())

    # training is completed, then evaluate model on the test set
    loss, accuracy = test(trained_model, test_loader)
    print(f"{loss = }")
    print(f"{accuracy = }")


model_1 = model_2 =  model_3 = model_4 = model_5 = SimpleNN(input_size, hidden_size, num_classes)
models = [model_1, model_2, model_3, model_4, model_5]
trained_weights = []
   


for i in range(num_subsets):
    print(f"Model {i+1}")
    run_model(epochs=5, lr=0.01, model=models[i], weights_list=trained_weights, train_dataset=subsets[i], test_dataset=test_dataset )
    


def average_model_weights(weight_list):
    """
    Average the weights of a list of models with the same architecture.

    Args:
        weight_list (list): List of PyTorch models weights

    Returns:
        dict: Averaged model weights.
    """
    # Initialize an empty dictionary to store the averaged weights
    avg_weights = {}

    # Get the state_dict of the first model weight as a template
    first_model_state_dict = weight_list[0]
    
    # Initialize the avg_weights with the structure of the first model
    for key in first_model_state_dict.keys():
        avg_weights[key] = torch.zeros_like(first_model_state_dict[key])
    
    # Iterate through each model and accumulate the weights
    for weight in weight_list:
        state_dict = weight
        for key in state_dict.keys():
            avg_weights[key] += state_dict[key]
    
    # Divide each weight by the number of models to get the average
    num_models = len(weight_list)
    for key in avg_weights.keys():
        avg_weights[key] /= num_models
    
    return avg_weights


averaged_weights = average_model_weights(weight_list=trained_weights)

new_model = SimpleNN(input_size, hidden_size, num_classes)  # Create the model with the same architecture
new_model.load_state_dict(averaged_weights)

loss, accuracy = test(model_1, DataLoader(test_dataset, batch_size=2, shuffle=False))
print(f"{loss = }")
print(f"{accuracy = }")





