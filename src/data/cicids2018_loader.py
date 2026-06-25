"""
CICIDS-2018 dataset loader for XAI-E-DiD
Loads and processes the CICIDS-2018 dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_cicids2018(data_dir: str = "data/cicids2018"):
    """
    Load CICIDS-2018 dataset.
    
    Args:
        data_dir: Directory containing CICIDS-2018 CSV files
    
    Returns:
        DataFrame with all data
    """
    data_path = Path(data_dir)
    
    # CICIDS-2018 files
    csv_files = [
        "Botnet.csv",
        "DDoS.csv",
        "DoS-GoldenEye.csv",
        "DoS-Hulk.csv",
        "DoS-Slowhttptest.csv",
        "DoS-Slowloris.csv",
        "FTP-BruteForce.csv",
        "Infilteration.csv",
        "SQL-Injection.csv",
        "SSH-Bruteforce.csv"
    ]
    
    dfs = []
    for csv_file in csv_files:
        file_path = data_path / csv_file
        if file_path.exists():
            print(f"Loading {csv_file}...")
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            print(f"Warning: {csv_file} not found")
    
    if not dfs:
        raise FileNotFoundError(f"No CICIDS-2018 files found in {data_dir}")
    
    # Combine all dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('.', '')
    
    # Handle missing values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Map labels to binary
    label_mapping = {
        'Benign': 0,
        'Bot': 1,
        'DDoS': 1,
        'DoS': 1,
        'FTP-BruteForce': 1,
        'Infilteration': 1,
        'SQL-Injection': 1,
        'SSH-Bruteforce': 1
    }
    
    if 'Label' in df.columns:
        df['label'] = df['Label'].map(label_mapping)
        df = df.drop(columns=['Label'])
    elif 'label' in df.columns:
        df['label'] = df['label'].str.strip().map(label_mapping)
    
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print(f"Loaded CICIDS-2018: {df.shape}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    return df


if __name__ == "__main__":
    df = load_cicids2018()
    print(df.head())
