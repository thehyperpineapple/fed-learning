import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import sys
sys.path.append('../../scripts')
from datasets_gen import CustomDataset
from simple_nn import SimpleNN, train, test, run_model

check_privacy = input("Do you want to cehck for privacy? (y/n) ")

df = pd.read_csv("../../datasets/gearbox_fault/gearbox_fault_data.csv", index_col=[0])
X = df.drop(columns= ['condition','encoded_condition'])
Y = df['encoded_condition']

print("Status: Splitting data")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape)

train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

input_size = X.shape[1]
hidden_size = 10
num_classes = 2

print("Status: Creating model")
model = SimpleNN(input_size, hidden_size, num_classes)
# model.cuda()
weight_list = []

print("Status: Training model")
run_model(epochs=20, lr = 0.01, model=model, weights_list=weight_list,train_dataset=train_dataset, test_dataset=test_dataset)

print("Status: Testing model\n")
if check_privacy=='y':
    metrics = test(model, DataLoader(train_dataset, batch_size=16, shuffle=False))
else:
    metrics = test(model, DataLoader(test_dataset, batch_size=16, shuffle=False))
print(f"loss = {metrics[0]} ")
print(f"accuracy = {metrics[1]}")
print(f"precision = {metrics[2]} ")
print(f"recall = {metrics[3]}")
print(f"f1 = {metrics[4]} ")