import torch
import pandas as pd
import numpy as np
from model import TabTransformer
from utils import set_seed
from process import process_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def submit_transformer():
    # Set seed and device
    set_seed()

    # Load sample submission and get target column names
    sample_submission = pd.read_csv('data/sample_submission.csv', index_col='id')
    target_cols = sample_submission.columns.tolist()

    # Load data
    X_cat_train, X_cont_train, y_train, X_cat_test, X_cont_test, id_map, input_dim = process_data(
        de_train_path='data/de_train.parquet',
        id_map_path='data/id_map.csv',
        isPCA=False
    )

    # Convert to torch tensors
    X_cat_test_tensor = torch.tensor(X_cat_test, dtype=torch.float32).to(device)
    X_cont_test_tensor = torch.tensor(X_cont_test, dtype=torch.float32).to(device)

    # Model parameters (must match training)
    model = TabTransformer(
        cat_input_dim=X_cat_test.shape[1],      # 152
        cont_input_dim=X_cont_test.shape[1],    # 56681
        emb_dim=128,
        num_heads=8,
        num_layers=1,
        output_dim=len(target_cols),            # 18211
        dropout=0.3
    )
    model.load_state_dict(torch.load('./output/transformer.pt', map_location=device))
    model.to(device)
    model.eval()

    # Inference
    with torch.no_grad():
        preds = model(X_cat_test_tensor, X_cont_test_tensor).cpu().numpy()

    # Save to submission.csv
    submission_df = pd.DataFrame(preds, index=id_map['id'].values, columns=target_cols)
    submission_df.index.name = 'id'
    submission_df.to_csv('./output/submission.csv')
    print("✅ Submission saved to submission.csv")
