# dataset.py

import os
import math
import random 
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class scPerturbDataset(Dataset):
    def __init__(self, X_cat, X_cont, y=None, model_type="MLP"):
        """
        X_cat: categorical features
        X_cont: continuous features
        y: targets, or None for test data
        model_type: determines the format for categorical features ("MLP" or "Transformer")
        """
        if model_type == "MLP":
            # MLP expects one-hot encoded categorical features
            self.X_cat = torch.tensor(X_cat, dtype=torch.float32)  # One-hot encoded
        elif model_type == "Transformer":
            # Transformer expects indices (integers)
            self.X_cat = torch.tensor(X_cat, dtype=torch.long)  # Indices for embedding
        else:
            raise ValueError("Invalid model type. Choose either 'MLP' or 'Transformer'.")
        
        self.X_cont = torch.tensor(X_cont, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_cat[idx], self.X_cont[idx], self.y[idx]
        else:
            return self.X_cat[idx], self.X_cont[idx]  # For test data


def create_dataloaders(X_cat_train, X_cont_train, y_train, X_cat_test=None, X_cont_test=None, batch_size=32, val_split=0.1, shuffle=True, model_type="MLP"):
    """
    Splits training data into train/val sets and returns DataLoaders for MLP or Transformer models.
    model_type: "MLP" or "Transformer" to choose the dataset format
    """
    dataset = scPerturbDataset(X_cat_train, X_cont_train, y_train, model_type=model_type)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    test_loader = None
    if X_cat_test is not None and X_cont_test is not None:
        test_dataset = scPerturbDataset(X_cat_test, X_cont_test, None, model_type=model_type)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

