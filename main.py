import os
import torch
import argparse
from process import process_data
from dataset import create_dataloaders
from train import run_training
from submit_mlp import submit_mlp
from submit_transformer import submit_transformer
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Run training pipeline")

    # Model and training configuration
    parser.add_argument('--model-type', type=str, choices=['MLP', 'Transformer'], default=config.MODEL_TYPE)
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--learning-rate', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--optimizer', type=str, default=config.OPTIMIZER)
    parser.add_argument('--scheduler', type=str, default=config.SCHEDULER)

    # Data paths
    parser.add_argument('--train-data', type=str, default=config.TRAIN_DATA_PATH)
    parser.add_argument('--id-map', type=str, default=config.ID_MAP_PATH)

    # run PCA
    parser.add_argument('--run-pca', action='store_true', help="Enable PCA on continuous features")

    # Optional control flags
    parser.add_argument('--no-train', action='store_true', help="Skip model training")
    parser.add_argument('--no-submit', action='store_true', help="Skip submission generation")

    return parser.parse_args()

def main(args):
    # Step 1: Load and process data
    X_cat_train, X_cont_train, y_train, X_cat_test, X_cont_test, id_map, input_dim = process_data(
        de_train_path=args.train_data,
        id_map_path=args.id_map,
        isPCA=args.run_pca                 # will be True only if --run-pca is passed, False if flag not passed
    )

    # Step 2: Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_cat_train, X_cont_train, y_train,
        X_cat_test=X_cat_test, X_cont_test=X_cont_test,
        batch_size=args.batch_size,
        model_type=args.model_type
    )

    # Step 3: Train model
    if not args.no_train:                   # will be True only if --no-train is passed, False if flag not passed
        run_training(
            X_cat_train, X_cont_train, y_train,
            X_cat_test, X_cont_test,
            model_type=args.model_type,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            optimizer_type=args.optimizer,
            scheduler_type=args.scheduler
        )

    # Step 4: Submit predictions
    if not args.no_submit:                  # will be True only if --no-submit is passed, False if flag not passed
        if args.model_type == "MLP":
            submit_mlp()
        elif args.model_type == "Transformer":
            submit_transformer()
        else:
            print("Unsupported model type for submission.")

if __name__ == "__main__":
    args = parse_args()
    main(args)