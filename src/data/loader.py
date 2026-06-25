"""
Data loader for XAI-E-DiD
Loads and merges CSV files from data/raw/ directory
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import glob


def load_data(data_dir: str = "data/raw", chunksize: Optional[int] = None) -> pd.DataFrame:
    """
    Load all CSV files from data/raw/ directory and merge into one DataFrame.
    
    Args:
        data_dir: Directory containing CSV files (default: "data/raw")
        chunksize: If specified, read files in chunks for memory efficiency
    
    Returns:
        DataFrame with numeric columns and label column (0=normal, 1=attack)
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Get all CSV files
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    print(f"Found {len(csv_files)} CSV file(s) in {data_dir}")
    
    # Load and merge CSV files
    dfs = []
    for csv_file in csv_files:
        print(f"Loading: {csv_file.name}")
        
        if chunksize:
            # Handle large files by reading in chunks
            chunks = []
            for chunk in pd.read_csv(csv_file, chunksize=chunksize, low_memory=False):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(csv_file, low_memory=False)
        
        dfs.append(df)
        print(f"  Shape: {df.shape}")
    
    # Merge all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"\nMerged dataset shape: {merged_df.shape}")
    
    # Keep only numeric columns
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    merged_df = merged_df[numeric_cols]
    print(f"Numeric columns: {len(numeric_cols)}")
    
    # Create label column if it doesn't exist
    if 'label' not in merged_df.columns:
        # Try to infer label from existing columns
        # Common column names for attack labels in intrusion detection datasets
        label_candidates = ['label', 'Label', 'class', 'Class', 'attack', 'Attack', 'outcome', 'Outcome']
        
        label_col = None
        for candidate in label_candidates:
            if candidate in merged_df.columns:
                label_col = candidate
                break
        
        if label_col:
            print(f"Found existing label column: {label_col}")
            # Convert to binary: 0 = normal, 1 = attack
            # Assuming normal traffic is labeled as 0 or 'normal'/'Normal'
            unique_values = merged_df[label_col].unique()
            print(f"Unique values in {label_col}: {unique_values}")
            
            # Convert to binary labels
            merged_df['label'] = merged_df[label_col].apply(lambda x: 0 if str(x).lower() in ['0', 'normal', 'benign'] else 1)
        else:
            # If no label column found, use last column as label
            print("No explicit label column found, using last column as label")
            last_col = merged_df.columns[-1]
            print(f"Using column: {last_col}")
            merged_df['label'] = merged_df[last_col].apply(lambda x: 0 if str(x).lower() in ['0', 'normal', 'benign'] else 1)
            merged_df = merged_df.drop(columns=[last_col])
    else:
        # Convert existing label column to binary
        print("Converting existing label column to binary (0=normal, 1=attack)")
        merged_df['label'] = merged_df['label'].apply(lambda x: 0 if str(x).lower() in ['0', 'normal', 'benign'] else 1)
    
    # Print final dataset info
    print(f"\nFinal dataset shape: {merged_df.shape}")
    print(f"Label distribution:")
    print(merged_df['label'].value_counts())
    print(f"Class imbalance ratio: {(merged_df['label'].value_counts()[0] / merged_df['label'].value_counts()[1]):.2f}:1")
    
    return merged_df


def load_data_efficient(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load data efficiently using dtype optimization for large files.
    
    Args:
        data_dir: Directory containing CSV files (default: "data/raw")
    
    Returns:
        DataFrame with numeric columns and label column (0=normal, 1=attack)
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    print(f"Found {len(csv_files)} CSV file(s) in {data_dir}")
    
    # Optimize dtypes for memory efficiency
    dtype_optimization = {
        'int64': 'int32',
        'float64': 'float32'
    }
    
    dfs = []
    for csv_file in csv_files:
        print(f"Loading: {csv_file.name}")
        
        # Read first chunk to infer dtypes
        sample_df = pd.read_csv(csv_file, nrows=1000, low_memory=False)
        
        # Optimize dtypes
        dtype_dict = {}
        for col in sample_df.columns:
            col_type = sample_df[col].dtype
            if col_type == 'int64':
                dtype_dict[col] = 'int32'
            elif col_type == 'float64':
                dtype_dict[col] = 'float32'
        
        # Read full data with optimized dtypes
        df = pd.read_csv(csv_file, dtype=dtype_dict, low_memory=False)
        
        dfs.append(df)
        print(f"  Shape: {df.shape}")
    
    # Merge all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"\nMerged dataset shape: {merged_df.shape}")
    
    # Keep only numeric columns
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    merged_df = merged_df[numeric_cols]
    print(f"Numeric columns: {len(numeric_cols)}")
    
    # Create label column
    if 'label' not in merged_df.columns:
        label_candidates = ['label', 'Label', 'class', 'Class', 'attack', 'Attack', 'outcome', 'Outcome']
        
        label_col = None
        for candidate in label_candidates:
            if candidate in merged_df.columns:
                label_col = candidate
                break
        
        if label_col:
            print(f"Found existing label column: {label_col}")
            merged_df['label'] = merged_df[label_col].apply(lambda x: 0 if str(x).lower() in ['0', 'normal', 'benign'] else 1)
        else:
            print("No explicit label column found, using last column as label")
            last_col = merged_df.columns[-1]
            print(f"Using column: {last_col}")
            merged_df['label'] = merged_df[last_col].apply(lambda x: 0 if str(x).lower() in ['0', 'normal', 'benign'] else 1)
            merged_df = merged_df.drop(columns=[last_col])
    else:
        print("Converting existing label column to binary (0=normal, 1=attack)")
        merged_df['label'] = merged_df['label'].apply(lambda x: 0 if str(x).lower() in ['0', 'normal', 'benign'] else 1)
    
    # Print final dataset info
    print(f"\nFinal dataset shape: {merged_df.shape}")
    print(f"Label distribution:")
    print(merged_df['label'].value_counts())
    print(f"Class imbalance ratio: {(merged_df['label'].value_counts()[0] / merged_df['label'].value_counts()[1]):.2f}:1")
    
    return merged_df


if __name__ == "__main__":
    # Example usage
    print("Loading data from data/raw/...")
    df = load_data_efficient("data/raw")
    print(f"\nDataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
