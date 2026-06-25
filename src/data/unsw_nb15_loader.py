"""
UNSW-NB15 dataset loader for XAI-E-DiD
Loads and processes the UNSW-NB15 dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_unsw_nb15(data_dir: str = "data/unsw_nb15"):
    """
    Load UNSW-NB15 dataset.
    
    Args:
        data_dir: Directory containing UNSW-NB15 CSV files
    
    Returns:
        DataFrame with all data
    """
    data_path = Path(data_dir)
    
    # UNSW-NB15 has training and test sets
    train_file = data_path / "UNSW_NB15_training-set.csv"
    test_file = data_path / "UNSW_NB15_testing-set.csv"
    
    dfs = []
    if train_file.exists():
        print(f"Loading {train_file.name}...")
        df_train = pd.read_csv(train_file)
        dfs.append(df_train)
    else:
        print(f"Warning: {train_file.name} not found")
    
    if test_file.exists():
        print(f"Loading {test_file.name}...")
        df_test = pd.read_csv(test_file)
        dfs.append(df_test)
    else:
        print(f"Warning: {test_file.name} not found")
    
    if not dfs:
        raise FileNotFoundError(f"No UNSW-NB15 files found in {data_dir}")
    
    # Combine dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('.', '')
    
    # Handle missing values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Map attack categories to binary
    # UNSW-NB15 has attack_cat column
    attack_categories = {
        'Normal': 0,
        'Analysis': 1,
        'Backdoor': 1,
        'DoS': 1,
        'Exploits': 1,
        'Fuzzers': 1,
        'Generic': 1,
        'Reconnaissance': 1,
        'Shellcode': 1,
        'Worms': 1
    }
    
    if 'attack_cat' in df.columns:
        df['label'] = df['attack_cat'].map(attack_categories)
    elif 'label' in df.columns:
        df['label'] = df['label'].map(attack_categories)
    
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print(f"Loaded UNSW-NB15: {df.shape}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    return df


if __name__ == "__main__":
    df = load_unsw_nb15()
    print(df.head())
