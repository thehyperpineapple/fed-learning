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

df = pd.read_csv("../../datasets/telemetry_analysis/processed/telemetry_analysis.csv", index_col=[0])
X = df.drop(columns= ['machineID','encoded_errors'])
Y = df['encoded_errors']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


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
