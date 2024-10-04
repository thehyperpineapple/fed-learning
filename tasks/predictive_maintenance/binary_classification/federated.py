import pandas as pd
from sklearn.model_selection import train_test_split
<<<<<<< HEAD
from torch.utils.data import DataLoader
import torch
import sys
sys.path.append('../../../scripts')
from datasets_gen import CustomDataset, create_subsets
from simple_nn import SimpleNN, train, test, run_model
from federated_functions import average_model_weights, fedprox_aggregate, scaffold_aggregate
from rich.console import Console
from rich.table import Table



import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../../../datasets/predictive_maintenance/balanced_pca_binary_classification.csv", index_col=[0])
X = df.drop(columns= ['Target'])
Y = df['Target']


print("Status: Splitting data")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

# Check for checking privacy
check = input('Do you want to check for privacy? (y/n) ')
if check == 'y':
    privacy_check = True
else:
    privacy_check = False

# Array of node numbers to iterate through
node_numbers = [1,5,10,25,50,100,200]  # Example array of numbers of nodes

# Loop through node numbers
for num_subsets in node_numbers:
    print(f"Running for {num_subsets} nodes")

    subsets = []
    subsets = create_subsets(subsets, num_subsets, train_dataset)

    input_size = X.shape[1]
    hidden_size = 25
    training_weights = []

    # Training each subset model
    for i in range(len(subsets)):
        # print(f"Status: Training model {i+1} with {num_subsets} nodes")
        model = SimpleNN(input_size, hidden_size)
        run_model(epochs=50, lr=0.01, model=model, weights_list=training_weights, train_dataset=subsets[i], test_dataset=test_dataset, batch_size=16)
        # print(f"Status: Completed training model {i+1}")

    # Averaging weights
    average_weights = average_model_weights(weight_list=training_weights)

    # Aggregating weights using FedProx
    new_model = SimpleNN(input_size, hidden_size)
    fedprox_model = fedprox_aggregate(global_model=new_model, local_models=training_weights, mu=1)
    new_model.load_state_dict(average_weights)

    # Testing models
    for model_name, model in [("FedAvg", new_model), ("FedProx", fedprox_model)]:
        # print(f"Status: Testing {model_name} model with {num_subsets} nodes")
        if privacy_check:
            metrics = test(model, DataLoader(train_dataset, batch_size=16, shuffle=False))
        else:
            metrics = test(model, DataLoader(test_dataset, batch_size=16, shuffle=False))

        # Display metrics
        # table = Table(title=f"Metrics of {model_name} model with {num_subsets} nodes")
        # rows = [[str(metrics[0]), str(metrics[1] * 100), str(metrics[2]), str(metrics[3]), str(metrics[4])]]
        # columns = ["Loss", "Accuracy", "Precision", "Recall", "F1 Score"]

        # for column in columns:
        #     table.add_column(column)

        # for row in rows:
        #     table.add_row(*row, style='bright_green')

        # console = Console()
        # console.print(table)
    print(str(metrics[0]), str(metrics[1] * 100), str(metrics[2]), str(metrics[3]), str(metrics[4]))
=======
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
>>>>>>> 56dae2d39cc93cd6b73324bcdf3e57c8b1f32ca0
