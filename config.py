# config.py

import torch


# General Settings
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data paths
TRAIN_DATA_PATH = 'data/de_train.parquet'
ID_MAP_PATH = 'data/id_map.csv'
SAMPLE_SUBMISSION_PATH = 'data/sample_submission.csv'

# Model Hyperparameters
MODEL_TYPE = "MLP"  # Options: "MLP", "Transformer"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
OPTIMIZER = "adam"
SCHEDULER = "plateau"

# Model-specific hyperparameters
MLP_HYPERPARAMS = {
    'input_dim': 1000,  # Example value, adjust based on your dataset
    'output_dim': 18211
}

TRANSFORMER_HYPERPARAMS = {
    'cat_input_dim': 152,  # Example value
    'cont_input_dim': 56681,  # Example value
    'emb_dim': 128,
    'num_heads': 8,
    'num_layers': 1,
    'output_dim': 18211,
    'dropout': 0.3
}
