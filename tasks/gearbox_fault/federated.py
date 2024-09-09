import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import sys
import torch
from rich.console import Console
from rich.table import Table

sys.path.append('../../scripts')
from datasets_gen import CustomDataset, create_subsets
from simple_nn import SimpleNN, RNN, train, test, run_model
from federated_functions import average_model_weights, fedprox_aggregate, scaffold_aggregate



import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../../datasets/gearbox_fault/gearbox_fault_data.csv", index_col=[0])
X = df.drop(columns= ['condition','encoded_condition'])
Y = df['encoded_condition']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

# Check for checking privacy
check = input('Do you want to check for privacy? (y/n) ')
if check == 'y':
    privacy_check = True
else:
    privacy_check = False

# Split the train_dataset into subsets
num_subsets = int(input("Enter number of nodes: "))
subsets = []

subsets = create_subsets(subsets, num_subsets, train_dataset)

input_size = X.shape[1]
hidden_size = 20
num_classes = 2

training_weights = []
# print(len(subsets[0]))
for i in range(len(subsets)):
    print(f"Status: Training model {i+1}")
    # model = SimpleNN(input_size, hidden_size, num_classes)
    model = SimpleNN(input_size, hidden_size, num_classes)
    # model.cuda()
    run_model(epochs=5, lr=0.01, model=model, weights_list=training_weights, train_dataset=subsets[i], test_dataset=test_dataset )
    print(f"Status: Completed training model {i+1} ")
    # torch.save(models[i].state_dict(), f"federated_models/federated_model_{i+1}.pt")

# len(training_weights)


average_weights = average_model_weights(weight_list=training_weights)
# Aggregating weights using FedProx

new_model = SimpleNN(input_size, hidden_size, num_classes)

# Aggregating weights using FedProx
fedprox_model = fedprox_aggregate(global_model=new_model, local_models=training_weights, mu=1)
# scaffold_model = scaffold_aggregate(global_model=new_model, local_models=training_weights, global_control_variate, local_control_variates)
new_model.load_state_dict(average_weights)


for model_name, model in [("FedAvg", new_model), ("FedProx", fedprox_model)]:
    print(f"Status: Testing {model_name} model with {num_subsets} nodes")
    if privacy_check == True:
        metrics = test(model, DataLoader(train_dataset, batch_size=16, shuffle=False))
    else:
        metrics = test(model, DataLoader(test_dataset, batch_size=16, shuffle=False))

    table = Table(title=f"Metrics of {model_name} model with {num_subsets} nodes")
    rows = [
        [str(metrics[0]),
        str(metrics[1]),
        str(metrics[2]),
        str(metrics[3]),
        str(metrics[4])],

    ]
    columns = ["Loss", "Accuracy", "Precision", "Recall", "F1 Score"]

    for column in columns:
        table.add_column(column)

    for row in rows:
        table.add_row(*row, style='bright_green')

    console = Console()
    console.print(table)

