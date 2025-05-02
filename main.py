import os
import torch
from process import process_data
from dataset import create_dataloaders
from model import MLP, TabTransformer
from train import run_training
from submit_mlp import submit_mlp
from submit_transformer import submit_transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(model_type="MLP", train_model=True, submit_model=True, epochs=10, batch_size=32, learning_rate=1e-3, optimizer_type="adam", scheduler_type="plateau"):
    # Step 1: Load and process data
    X_cat_train, X_cont_train, y_train, X_cat_test, X_cont_test, id_map, input_dim = process_data(isPCA=False)

    # Step 2: Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(X_cat_train, X_cont_train, y_train, 
                                                               X_cat_test=X_cat_test, X_cont_test=X_cont_test, 
                                                               batch_size=batch_size, model_type=model_type)

    # Step 3: Train model
    if train_model:
        run_training(X_cat_train, X_cont_train, y_train, X_cat_test, X_cont_test,
                     model_type=model_type, batch_size=batch_size, epochs=epochs,
                     learning_rate=learning_rate, optimizer_type=optimizer_type, scheduler_type=scheduler_type)
    
    # Step 4: Submit predictions
    if submit_model:
        if model_type == "MLP":
            submit_mlp()  # This assumes submit_mlp.py contains the submit logic for MLP
        elif model_type == "Transformer":
            submit_transformer()  # This assumes submit_transformer.py contains the submit logic for Transformer
        else:
            print("Unsupported model type for submission.")

if __name__ == "__main__":
    # Change the model type to either "MLP" or "Transformer"
    model_type = "MLP"  
    # model_type = "Transformer"  
    
    main(model_type=model_type, train_model=True, submit_model=True, epochs=10)
