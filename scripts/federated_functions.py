from torch.utils.data import Subset
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

def average_model_weights(weight_list):
    """
    Average the weights of a list of models with the same architecture.

    Args:
        models (list): List of PyTorch models with the same architecture.

    Returns:
        dict: Averaged model weights.
    """
    # Initialize an empty dictionary to store the averaged weights
    avg_weights = {}

    # Get the state_dict of the first model as a template
    first_model_state_dict = weight_list[0]
    
    # Initialize the avg_weights with the structure of the first model
    for key in first_model_state_dict.keys():
        avg_weights[key] = torch.zeros_like(first_model_state_dict[key])
    
    # Iterate through each model and accumulate the weights
    for weight in weight_list:
        state_dict = weight
        for key in state_dict.keys():
            avg_weights[key] += state_dict[key]
    
    # Divide each weight by the number of models to get the average
    num_models = len(weight_list)
    for key in avg_weights.keys():
        avg_weights[key] /= num_models
    
    return avg_weights


def create_subsets(num_subsets, train_dataset, subsets):
    # Split the train_dataset into 5 subsets
    subset_size = len(train_dataset) // num_subsets
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)

    for i in range(num_subsets):
        start_idx = i * subset_size
        end_idx = (i + 1) * subset_size if i != num_subsets - 1 else len(train_dataset)
        subset_indices = indices[start_idx:end_idx]
        subsets.append(Subset(train_dataset, subset_indices))

def fedprox_aggregate(global_model, local_models, mu=0.01):
    avg_model = global_model
    for key in avg_model.state_dict().keys():
        avg_model.state_dict()[key] = torch.zeros_like(avg_model.state_dict()[key])
        for local_model in local_models:
            avg_model.state_dict()[key] += (1 - mu) * local_model.state_dict()[key]
        avg_model.state_dict()[key] += mu * global_model.state_dict()[key]
    avg_model.state_dict()[key] = torch.div(avg_model.state_dict()[key], len(local_models))
    return avg_model


def scaffold_aggregate(global_model, local_models, global_control_variate, local_control_variates):
    avg_model = global_model
    for key in avg_model.state_dict().keys():
        avg_model.state_dict()[key] = torch.zeros_like(avg_model.state_dict()[key])
        for i in range(len(local_models)):
            control_delta = local_control_variates[i].state_dict()[key] - global_control_variate.state_dict()[key]
            avg_model.state_dict()[key] += local_models[i].state_dict()[key] - control_delta
        avg_model.state_dict()[key] = torch.div(avg_model.state_dict()[key], len(local_models))
    return avg_model
