import pandas as pd
from sklearn.model_selection import train_test_split
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