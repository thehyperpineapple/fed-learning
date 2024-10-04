import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import sys
sys.path.append('../../scripts')
from datasets_gen import CustomDataset
<<<<<<< HEAD
from simple_nn import SimpleNN, test, run_model
from machine_learning import logistic_regression_pipeline
from rich.console import Console
from rich.table import Table

check_privacy = input("Do you want to check for privacy? (y/n) ")


df = pd.read_csv("../../datasets/gearbox_fault/balanced_gearbox_fault_data.csv", index_col=[0])
X = df.drop(columns= ['encoded_condition'])
=======
from simple_nn import SimpleNN, train, test, run_model

check_privacy = input("Do you want to cehck for privacy? (y/n) ")

df = pd.read_csv("../../datasets/gearbox_fault/gearbox_fault_data.csv", index_col=[0])
X = df.drop(columns= ['condition','encoded_condition'])
>>>>>>> 56dae2d39cc93cd6b73324bcdf3e57c8b1f32ca0
Y = df['encoded_condition']

print("Status: Splitting data")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
<<<<<<< HEAD
=======
print(x_train.shape, y_train.shape)
>>>>>>> 56dae2d39cc93cd6b73324bcdf3e57c8b1f32ca0

train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

input_size = X.shape[1]
<<<<<<< HEAD
hidden_size = 20

print("Status: Creating model")
model = SimpleNN(input_size, hidden_size)

weight_list = []

print("Status: Training models")
# run_model(epochs=5, lr = 0.01, model=model, weights_list=weight_list,train_dataset=train_dataset, test_dataset=test_dataset, batch_size=32)

print("Status: Testing model\n")
if check_privacy=='y':
    # metrics_nn = test(model, DataLoader(train_dataset, batch_size=32, shuffle=False))
    # metrics_lr = logistic_regression_pipeline(x_train,y_train,x_test,y_test,check_privacy=True, federated=False)
    metrics_lr_fed = logistic_regression_pipeline(x_train,y_train,x_test,y_test,check_privacy=True, federated=True)
else:
    # metrics_nn = test(model, DataLoader(test_dataset, batch_size=32, shuffle=False))
    # metrics_lr = logistic_regression_pipeline(x_train,y_train,x_test,y_test,check_privacy=False, federated=False)
    metrics_lr_fed = logistic_regression_pipeline(x_train,y_train,x_test,y_test,check_privacy=False, federated=True)


table = Table(title="Metrics")

rows = [
    # ["ANN", str(metrics_nn[1]*100), str(metrics_nn[2]), str(metrics_nn[3]), str(metrics_nn[4])],
    # ["Logistic Regression", str(metrics_lr[0]*100), str(metrics_lr[1]), str(metrics_lr[2]), str(metrics_lr[3])],    
    ["Logistic Regression - Federated", str(metrics_lr_fed[0]*100), str(metrics_lr_fed[1]), str(metrics_lr_fed[2]), str(metrics_lr_fed[3])]
]

columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]

for column in columns:
    table.add_column(column)

for row in rows:
    table.add_row(*row, style='bright_green')

console = Console()
console.print(table)
=======
hidden_size = 10
num_classes = 1

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
>>>>>>> 56dae2d39cc93cd6b73324bcdf3e57c8b1f32ca0
