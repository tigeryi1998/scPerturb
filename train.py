# train.py

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

from sklearn.metrics import mean_squared_error
from model import MLP, TabTransformer
from dataset import create_dataloaders
from utils import set_seed, get_optimizer, get_scheduler, plot_loss, MRRMSLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directory if it doesn't exist
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# Training loop
def train(model, train_loader, val_loader, criterion, optimizer, epochs=10, scheduler=None):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Save the model based on type
    if isinstance(model, MLP):
        model_name = os.path.join(output_dir, "mlp.pt")
    elif isinstance(model, TabTransformer):
        model_name = os.path.join(output_dir, "transformer.pt")
    else:
        model_name = os.path.join(output_dir, "best_model.pt")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (x_cat, x_cont, y) in enumerate(train_loader):
            x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(x_cat, x_cont)
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_cat, x_cont, y in val_loader:
                x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)
                outputs = model(x_cat, x_cont)
                loss = criterion(outputs, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if scheduler:
            scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Save model if it's the best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_name)
            print("Model saved!")

    # Plotting the losses
    loss_plot_path = os.path.join(output_dir, "loss_plot.png")
    plot_loss(train_losses, val_losses,save_path=loss_plot_path)


# Main function to handle dataset loading, model creation, and training
def run_training(X_cat_train, X_cont_train, y_train, X_cat_test=None, X_cont_test=None, 
                 model_type="MLP", batch_size=32, epochs=10, learning_rate=1e-3, optimizer_type="adamw", scheduler_type="plateau"):
    # Set random seed for reproducibility
    set_seed()

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(X_cat_train, X_cont_train, y_train, 
                                                               X_cat_test=X_cat_test, X_cont_test=X_cont_test, 
                                                               batch_size=batch_size, model_type=model_type)

    # Model initialization
    if model_type == "MLP":
        model = MLP(input_dim=X_cat_train.shape[1] + X_cont_train.shape[1], output_dim=y_train.shape[1]).to(device)
    elif model_type == "Transformer":
        model = TabTransformer(cat_input_dim=X_cat_train.shape[1], cont_input_dim=X_cont_train.shape[1],
                               output_dim=y_train.shape[1]).to(device)
    else:
        raise ValueError("Invalid model type. Choose either 'MLP' or 'Transformer'.")

    # Define loss function
    # criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
    criterion = MRRMSLoss()     # row wirse root mean square error self-defined 

    # Get optimizer and scheduler
    optimizer = get_optimizer(model, optimizer_name=optimizer_type, lr=learning_rate)
    scheduler = get_scheduler(optimizer, scheduler_type=scheduler_type)

    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, epochs=epochs, scheduler=scheduler)

    # Test the model (optional)
    if test_loader:
        predictions = []
        actuals = []
        with torch.no_grad():
            for x_cat, x_cont, y in test_loader:
                x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)
                outputs = model(x_cat, x_cont)
                predictions.append(outputs.cpu().numpy())
                actuals.append(y.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        test_mse = mean_squared_error(actuals, predictions)
        print(f"Test MSE: {test_mse:.4f}")

