import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append('../../scripts')
from datasets_gen import CustomDataset
from simple_nn import SimpleNN, train, test, run_model


df = pd.read_csv("../../datasets/multi_class_classification.csv", index_col=[0])


X = df.drop(columns='Failure Type')
Y = df['Failure Type']


# Perform one-hot encoding to Y
le = LabelEncoder()
le.fit(Y)
Y_transformed = le.fit_transform(Y)
Y_transformed = pd.Series(Y_transformed)

x_train, x_test, y_train, y_test = train_test_split(X, Y_transformed, test_size=0.2, random_state=42)


train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

# Creating Model

input_size = X.shape[1]
hidden_size = 10
num_classes = 6

model = SimpleNN(input_size, hidden_size, num_classes)
weight_list = []

run_model(epochs=5, lr = 0.01, model=model, weights_list=weight_list,train_dataset=train_dataset, test_dataset=test_dataset)


