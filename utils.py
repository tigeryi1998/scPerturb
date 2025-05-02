# utils.py

import os
import math
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=1):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return print(f"Random seed set to {seed}")


# NumPy version of Mean Row-wise RMSE
def mrrmse_np(y_true, y_pred):
    return np.mean(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1)))

# PyTorch version
def mrrmse_torch(y_true, y_pred):
    return torch.mean(torch.sqrt(torch.mean((y_true - y_pred) ** 2, dim=1)))

# Optional loss class for use with nn.Module
class MRRMSLoss(nn.Module):
    def __init__(self):
        super(MRRMSLoss, self).__init__()
    def forward(self, y_pred, y_true):
        return mrrmse_torch(y_true, y_pred)
    
# Plot loss functions
def plot_loss(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss over epochs.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker='.-')
    plt.plot(val_losses, label="Validation Loss", marker='.-')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


# get an optimizer
def get_optimizer(model, optimizer_name="adam", lr=1e-3, weight_decay=0.0):
    """
    Returns a PyTorch optimizer for the given model.
    
    Supported: 'adam', 'adamw', 'sgd', 'rmsprop'
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


# change scheduler
def get_scheduler(optimizer, scheduler_type="plateau", **kwargs):
    """
    Create a learning rate scheduler.
    Supported: 'plateau', 'step', 'multistep'
    """
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True, **kwargs
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1, **kwargs
        )
    elif scheduler_type == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 20], gamma=0.1, **kwargs
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
    

