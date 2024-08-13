import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import sys
sys.path.append('../../scripts')
from datasets_gen import CustomDataset
from simple_nn import SimpleNN, train, test, run_model


df = pd.read_csv("../../datasets/telemetry_analysis/processed/telemetry_analysis.csv", index_col=[0])


X = df.drop(columns= ['machineID', 'encoded_errors'])
Y = df['encoded_errors']


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape)

train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

input_size = X.shape[1]
hidden_size = 10
num_classes = 2


model = SimpleNN(input_size, hidden_size, num_classes)
weight_list = []

run_model(epochs=5, lr = 0.01, model=model, weights_list=weight_list,train_dataset=train_dataset, test_dataset=test_dataset)


torch.save(model.state_dict(), 'conventional.pt')


