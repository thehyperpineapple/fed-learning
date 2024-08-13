import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import sys
import torch
sys.path.append('../../scripts')
from datasets_gen import CustomDataset
from simple_nn import SimpleNN, train, test, run_model
from federated_functions import average_model_weights


df = pd.read_csv("../../datasets/telemetry_analysis/processed/telemetry_analysis.csv", index_col=[0])


X = df.drop(columns= ['encoded_errors'])
Y = df['encoded_errors']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

train = pd.concat([x_train, y_train], axis=1)

# Create a list to hold the DataFrame parts
df_parts = []

# Define the ranges
ranges = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100)]

# Loop through the ranges and slice the DataFrame
for start, end in ranges:
    df_part = train[(train['machineID'] >= start) & (train['machineID'] <= end)]
    df_parts.append(df_part)

df_processed =[]

for df in range(len(df_parts)):
    X = df_parts[df].drop(columns= ['machineID', 'encoded_errors'])
    X.shape
    Y = df_parts[df].encoded_errors
    print(X.shape, Y.shape)
    print(Y.iloc[0])
    train_dataset = CustomDataset(X, Y)


    df_processed.append(train_dataset)
    
x_test = x_test.drop(columns= ['machineID'])
test_dataset = CustomDataset(x_test, y_test)

input_size = X.shape[1]
hidden_size = 10
num_classes = 2

model_1 = model_2 =  model_3 = model_4 = model_5 = SimpleNN(input_size, hidden_size, num_classes)
models = [model_1, model_2, model_3, model_4, model_5]

training_weights = []


len(df_processed)


for i in range(len(df_processed)):
    print(f"Model {i+1}")
    run_model(epochs=5, lr=0.01, model=models[i], weights_list=training_weights, train_dataset=df_processed[i], test_dataset=test_dataset )
    torch.save(models[i].state_dict(), f"federated_models/federated_model_{i+1}.pt")


len(training_weights)


average_weights = average_model_weights(weight_list=training_weights)
new_model = SimpleNN(input_size, hidden_size, num_classes)
new_model.load_state_dict(average_weights)

metrics = test(new_model, DataLoader(test_dataset, batch_size=2, shuffle=False))
print(f"loss = {metrics[0]} ")
print(f"accuracy = {metrics[1]}")
print(f"precision = {metrics[2]} ")
print(f"recall = {metrics[3]}")
print(f"f1 = {metrics[4]} ")
