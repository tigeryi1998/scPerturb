# model.py

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

# input_dim = 56833 if isPCA==False
# input_dim = 512 if isPCA==True

class MLP(nn.Module):
    def __init__(self, input_dim=56833, hidden_dims=[4096, 2048, 1024, 512], output_dim=18211, dropout=0.3):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim

        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class TabTransformer(nn.Module):
    def __init__(self, cat_input_dim=152, cont_input_dim=56681, emb_dim=128, num_heads=8, num_layers=1, output_dim=18211, dropout=0.3):
        super(TabTransformer, self).__init__()

        # Linear projection of one-hot categorical features to embedding space
        self.cat_proj = nn.Linear(cat_input_dim, emb_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Continuous branch (optional projection)
        self.cont_proj = nn.Sequential(
            nn.LayerNorm(cont_input_dim),
            nn.Linear(cont_input_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(2 * emb_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x_cat, x_cont):
        # x_cat shape: (batch_size, cat_input_dim)
        # x_cont shape: (batch_size, cont_input_dim)

        # Process categorical data
        cat_emb = self.cat_proj(x_cat).unsqueeze(1)  # (batch_size, 1, emb_dim)
        cat_encoded = self.transformer_encoder(cat_emb).squeeze(1)  # (batch_size, emb_dim)

        # Process continuous data
        cont_emb = self.cont_proj(x_cont)  # (batch_size, emb_dim)

        # Combine both embeddings
        combined = torch.cat([cat_encoded, cont_emb], dim=1)  # (batch_size, 2 * emb_dim)
        
        # Pass through final fully connected layers
        out = self.fc(combined)

        return out
