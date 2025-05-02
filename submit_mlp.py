import torch
import pandas as pd
import numpy as np
from model import MLP
from process import process_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def submit_mlp():
    # Load data
    X_cat_train, X_cont_train, y_train, X_cat_test, X_cont_test, id_map, input_dim = process_data(isPCA=False)

    # Concatenate categorical and continuous test features
    X_test = np.hstack([X_cat_test, X_cont_test])
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Load trained model
    model = MLP(input_dim=X_test.shape[1], output_dim=y_train.shape[1])
    model.load_state_dict(torch.load("./output/mlp.pt", map_location=device))
    model.to(device)
    model.eval()

    # Predict
    with torch.no_grad():
        preds = model(X_test_tensor).cpu().numpy()

    # Format predictions to match sample_submission
    sample_submission = pd.read_csv("data/sample_submission.csv", index_col="id")
    submission = pd.DataFrame(preds, columns=sample_submission.columns)
    submission.index.name = "id"

    # Save submission file
    submission.to_csv("./output/submission_mlp.csv")
    print("✅ Submission saved to submission_mlp.csv")

