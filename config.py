# config.py

import torch

# General Settings
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data paths
TRAIN_DATA_PATH = './data/de_train.parquet'
ID_MAP_PATH = './data/id_map.csv'
SAMPLE_SUBMISSION_PATH = './data/sample_submission.csv'

# Model Hyperparameters
MODEL_TYPE = "MLP"  # Options: "MLP", "Transformer"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
OPTIMIZER = "adamw"
SCHEDULER = "plateau"
