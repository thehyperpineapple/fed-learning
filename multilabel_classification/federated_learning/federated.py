import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append('../../scripts')
from datasets_gen import CustomDataset
from simple_nn import SimpleNN, train, test, run_model
from federated_functions import average_model_weights, create_subsets


df = pd.read_csv("../../datasets/multi_class_classification.csv", index_col=[0])


X = df.drop(columns='Failure Type')
Y = df['Failure Type']


# Perform one-hot encoding to Y
le = LabelEncoder()
le.fit(Y)
Y_transformed = le.fit_transform(Y)
Y_transformed = pd.Series(Y_transformed)


print("Classes:", le.classes_)
print("Encoded values:", le.transform(le.classes_))


x_train, x_test, y_train, y_test = train_test_split(X, Y_transformed, test_size=0.2, random_state=42)


train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

# # Creating Models


input_size = X.shape[1]
hidden_size = 10
num_classes = 6

model_1 = model_2 =  model_3 = model_4 = model_5 = SimpleNN(input_size, hidden_size, num_classes)
models = [model_1, model_2, model_3, model_4, model_5]
trained_weights = []


subsets = []
num_subsets = 5
create_subsets(num_subsets=5, train_dataset=train_dataset, subsets=subsets)


for i in range(num_subsets):
    print(f"Model {i+1}")
    run_model(epochs=5, lr=0.01, model=models[i], weights_list=trained_weights, train_dataset=subsets[i], test_dataset=test_dataset )


average_weights = average_model_weights(weight_list=trained_weights)
new_model = SimpleNN(input_size, hidden_size, num_classes)
new_model.load_state_dict(average_weights)

loss, accuracy = test(model_1, DataLoader(test_dataset, batch_size=2, shuffle=False))
print(f"{loss = }")
print(f"{accuracy = }")





