import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import sys
import torch
sys.path.append('../../scripts')
from datasets_gen import CustomDataset, create_subsets
from simple_nn import SimpleNN, train, test, run_model
from federated_functions import average_model_weights

import warnings
warnings.filterwarnings('ignore')

<<<<<<< HEAD
df = pd.read_csv("../../datasets/telemetry_analysis/processed/balanced_telemetry_analysis.csv", index_col=[0])
=======
df = pd.read_csv("../../datasets/telemetry_analysis/processed/telemetry_analysis.csv", index_col=[0])
>>>>>>> 56dae2d39cc93cd6b73324bcdf3e57c8b1f32ca0
X = df.drop(columns= ['machineID','encoded_errors'])
Y = df['encoded_errors']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

<<<<<<< HEAD
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
    hidden_size = 20
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
    # fedprox_model = fedprox_aggregate(global_model=new_model, local_models=training_weights, mu=1)
    new_model.load_state_dict(average_weights)

    # Testing models
    for model_name, model in [("FedAvg", new_model)]:
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

train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

# Create a list to hold the DataFrame parts

# Split the train_dataset into 5 subsets
num_subsets = int(input("Enter number of nodes: "))
subsets = []

subsets = create_subsets(subsets, num_subsets, train_dataset)

input_size = X.shape[1]
hidden_size = 10
num_classes = 2

training_weights = []
print(len(subsets[0]))
for i in range(len(subsets)):
    print(f"Training Model {i+1}")
    model = SimpleNN(input_size, hidden_size, num_classes)
    run_model(epochs=5, lr=0.01, model=model, weights_list=training_weights, train_dataset=subsets[i], test_dataset=test_dataset )
    print(f"Finished training model {i+1} \n")
    # torch.save(models[i].state_dict(), f"federated_models/federated_model_{i+1}.pt")

# len(training_weights)


average_weights = average_model_weights(weight_list=training_weights)
new_model = SimpleNN(input_size, hidden_size, num_classes)
new_model.load_state_dict(average_weights)

metrics = test(new_model, DataLoader(test_dataset, batch_size=2, shuffle=False))
print(f"loss = {metrics[0]} ")
print(f"accuracy = {metrics[1]}")
print(f"precision = {metrics[2]} ")
print(f"recall = {metrics[3]}")
print(f"f1 = {metrics[4]} ")
>>>>>>> 56dae2d39cc93cd6b73324bcdf3e57c8b1f32ca0
