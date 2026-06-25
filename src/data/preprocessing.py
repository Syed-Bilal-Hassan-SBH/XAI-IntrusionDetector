"""
Preprocessing for XAI-E-DiD
Fills missing values with median, normalizes with MinMaxScaler, removes outliers by z-score, one-hot encoding
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_data(
    df: pd.DataFrame,
    save_scaler: bool = True,
    scaler_path: str = "models/scaler.pkl",
    z_threshold: float = 3.0,
    categorical_cols: list = None,
    one_hot_encode: bool = True
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Preprocess data: fill missing, remove outliers, normalize, one-hot encode.
    
    Args:
        df: Input DataFrame
        save_scaler: Whether to save the fitted scaler
        scaler_path: Path to save scaler
        z_threshold: Z-score threshold for outlier removal
        categorical_cols: List of categorical column names
        one_hot_encode: Whether to apply one-hot encoding
    
    Returns:
        Tuple of (preprocessed DataFrame, fitted scaler)
    """
    print("=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)
    print(f"Original shape: {df.shape}")
    
    # Auto-detect categorical columns if not provided
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Detected categorical columns: {categorical_cols}")
    
    # One-hot encode categorical columns
    if one_hot_encode and len(categorical_cols) > 0:
        print("Applying one-hot encoding...")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"Shape after one-hot encoding: {df.shape}")
    
    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols]
    print(f"Numeric columns: {len(numeric_cols)}")
    
    # Separate label column if exists
    label_col = None
    if 'label' in df.columns:
        label_col = df['label'].values
        df = df.drop(columns=['label'])
    
    # Fill missing values with median
    print("Filling missing values with median...")
    df = df.fillna(df.median())
    
    # Remove outliers using z-score
    print(f"Removing outliers with z-score threshold {z_threshold}...")
    z_scores = np.abs((df - df.mean()) / df.std())
    df = df[(z_scores < z_threshold).all(axis=1)]
    print(f"Shape after outlier removal: {df.shape}")
    
    # Normalize with MinMaxScaler
    print("Normalizing with MinMaxScaler...")
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    
    # Save scaler
    if save_scaler:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
    
    # Add label back if it was removed
    if label_col is not None:
        df_normalized['label'] = label_col
    
    print(f"Final shape: {df_normalized.shape}")
    print("=" * 60)
    
    return df_normalized, scaler


def apply_preprocessing(
    df: pd.DataFrame,
    scaler_path: str = "models/scaler.pkl",
    z_threshold: float = 3.0,
    categorical_cols: list = None,
    one_hot_encode: bool = True
) -> pd.DataFrame:
    """
    Apply preprocessing to new data using saved scaler.
    
    Args:
        df: Input DataFrame
        scaler_path: Path to saved scaler
        z_threshold: Z-score threshold for outlier removal
        categorical_cols: List of categorical column names
        one_hot_encode: Whether to apply one-hot encoding
    
    Returns:
        Preprocessed DataFrame
    """
    # Auto-detect categorical columns if not provided
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # One-hot encode categorical columns
    if one_hot_encode and len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols]
    
    # Separate label column if exists
    label_col = None
    if 'label' in df.columns:
        label_col = df['label'].values
        df = df.drop(columns=['label'])
    
    # Fill missing values with median
    df = df.fillna(df.median())
    
    # Remove outliers using z-score
    z_scores = np.abs((df - df.mean()) / df.std())
    df = df[(z_scores < z_threshold).all(axis=1)]
    
    # Normalize with saved scaler
    scaler = joblib.load(scaler_path)
    df_normalized = pd.DataFrame(
        scaler.transform(df),
        columns=df.columns,
        index=df.index
    )
    
    # Add label back if it was removed
    if label_col is not None:
        df_normalized['label'] = label_col
    
    print(f"Preprocessed shape: {df_normalized.shape}")
    
    return df_normalized


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("..")
    from loader import load_data_efficient
    
    df = load_data_efficient("data/raw")
    df_clean, scaler = preprocess_data(df)
    print(df_clean.head())
