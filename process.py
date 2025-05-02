# process.py

import os
import math
import random 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_csv(file_path):
    return pd.read_csv(file_path)

def read_parquet(file_path):
    return pd.read_parquet(file_path)

def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    return np.asarray(fp)

def process_data(de_train_path='data/de_train.parquet', id_map_path='data/id_map.csv', isPCA=False):
    df_train = pd.read_parquet(de_train_path)
    id_map = pd.read_csv(id_map_path)

    gene_columns = df_train.columns.difference(['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control'])
    y_train = df_train[gene_columns].copy()

    df_train['is_test'] = False
    id_map['is_test'] = True
    combined = pd.concat([df_train[['cell_type', 'sm_name', 'is_test']], 
                          id_map[['cell_type', 'sm_name', 'is_test']]], ignore_index=True)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(combined[['cell_type', 'sm_name']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['cell_type', 'sm_name']))
    combined_encoded = pd.concat([encoded_df, combined['is_test'].reset_index(drop=True)], axis=1)

    encoded_train = combined_encoded[combined_encoded['is_test'] == False].drop(columns='is_test').reset_index(drop=True)
    encoded_test = combined_encoded[combined_encoded['is_test'] == True].drop(columns='is_test').reset_index(drop=True)

    gene_stats_mean = df_train.groupby('sm_name')[gene_columns].mean()
    gene_stats_std = df_train.groupby('sm_name')[gene_columns].std()
    gene_stats_median = df_train.groupby('sm_name')[gene_columns].median()

    stats_train = df_train[['sm_name']].copy()
    stats_train_mean = stats_train['sm_name'].map(gene_stats_mean.to_dict(orient='index')).apply(pd.Series)
    stats_train_std = stats_train['sm_name'].map(gene_stats_std.to_dict(orient='index')).apply(pd.Series)
    stats_train_median = stats_train['sm_name'].map(gene_stats_median.to_dict(orient='index')).apply(pd.Series)
    stats_train_full = pd.concat([stats_train_mean, stats_train_std, stats_train_median], axis=1).fillna(0)

    smiles_list_train = df_train['SMILES'].apply(smiles_to_morgan)
    smiles_array_train = np.stack(smiles_list_train.to_numpy())

    X_cat_train = encoded_train.values
    X_cont_train = np.hstack([stats_train_full.values, smiles_array_train])

    df_train_unique = df_train.drop_duplicates(subset='sm_name')
    id_map['SMILES'] = id_map['sm_name'].map(df_train_unique.set_index('sm_name')['SMILES'])

    stats_test = id_map[['sm_name']].copy()
    stats_test_mean = stats_test['sm_name'].map(gene_stats_mean.to_dict(orient='index')).apply(pd.Series)
    stats_test_std = stats_test['sm_name'].map(gene_stats_std.to_dict(orient='index')).apply(pd.Series)
    stats_test_median = stats_test['sm_name'].map(gene_stats_median.to_dict(orient='index')).apply(pd.Series)
    stats_test_full = pd.concat([stats_test_mean, stats_test_std, stats_test_median], axis=1).fillna(0)

    smiles_list_test = id_map['SMILES'].apply(smiles_to_morgan)
    smiles_array_test = np.stack(smiles_list_test.to_numpy())

    X_cat_test = encoded_test.values
    X_cont_test = np.hstack([stats_test_full.values, smiles_array_test])

    if isPCA:
        pca_components = 512 
        pca = PCA(n_components=pca_components)
        X_combined = np.vstack([X_cont_train, X_cont_test])
        X_combined = pca.fit_transform(X_combined)
        X_cont_train = X_combined[:len(X_cont_train)]
        X_cont_test = X_combined[len(X_cont_train):]

    input_dim = X_cat_train.shape[1]

    return (X_cat_train.astype(np.float32), 
            X_cont_train.astype(np.float32), 
            y_train.astype(np.float32).values, 
            X_cat_test.astype(np.float32), 
            X_cont_test.astype(np.float32), 
            id_map, 
            input_dim
            )

if __name__ == "__main__":
    X_cat_train, X_cont_train, y_train, X_cat_test, X_cont_test, id_map, input_dim = process_data(isPCA=False)

    print("Shapes:")
    print(f"X_cat_train: {X_cat_train.shape}")
    print(f"X_cont_train: {X_cont_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_cat_test: {X_cat_test.shape}")
    print(f"X_cont_test: {X_cont_test.shape}")
    print(f"id_map: {id_map.shape}")
    print(f"input_dim (categorical features): {input_dim}")

    print("\nFirst 3 rows of X_cat_train:")
    print(X_cat_train[:3])

    print("\nFirst 3 rows of X_cont_train:")
    print(X_cont_train[:3])