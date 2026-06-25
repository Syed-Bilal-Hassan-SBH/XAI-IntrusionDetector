"""
CICIDS-2017 dataset loader for XAI-E-DiD
Loads and processes the CICIDS-2017 dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_cicids2017(data_dir: str = "data/cicids2017"):
    """
    Load CICIDS-2017 dataset.
    
    Args:
        data_dir: Directory containing CICIDS-2017 CSV files
    
    Returns:
        DataFrame with all data
    """
    data_path = Path(data_dir)
    
    # CICIDS-2017 has multiple CSV files for different days
    csv_files = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
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
        raise FileNotFoundError(f"No CICIDS-2017 files found in {data_dir}")
    
    # Combine all dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    # Clean column names (remove spaces, special characters)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('.', '')
    
    # Handle missing values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Map labels to binary (normal vs attack)
    # CICIDS-2017 has specific attack types
    label_mapping = {
        'BENIGN': 0,
        'DDoS': 1,
        'DoS Hulk': 1,
        'DoS GoldenEye': 1,
        'DoS Slowloris': 1,
        'DoS Slowhttptest': 1,
        'PortScan': 1,
        'Bot': 1,
        'Infiltration': 1,
        'Web Attack Brute Force': 1,
        'Web Attack Sql Injection': 1,
        'Web Attack XSS': 1,
        'FTP-Patator': 1,
        'SSH-Patator': 1,
        'Heartbleed': 1
    }
    
    # Normalize label column name
    if 'Label' in df.columns:
        df['label'] = df['Label'].map(label_mapping)
        df = df.drop(columns=['Label'])
    elif 'label' in df.columns:
        df['label'] = df['label'].str.upper().map(label_mapping)
    
    # Remove rows where label mapping failed
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print(f"Loaded CICIDS-2017: {df.shape}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    return df


def get_attack_category(sample: pd.Series) -> str:
    """
    Get attack category from sample.
    
    Args:
        sample: Sample row from dataset
    
    Returns:
        Attack category string
    """
    # This would be implemented based on specific features
    # For now, return generic categories
    return "Anomaly"


if __name__ == "__main__":
    # Test loader
    df = load_cicids2017()
    print(df.head())
