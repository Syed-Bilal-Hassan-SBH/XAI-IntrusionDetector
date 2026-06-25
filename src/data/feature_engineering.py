"""
Feature engineering module for XAI-E-DiD
Converts data into sequences for LSTM processing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
import joblib


def create_sequences(df: pd.DataFrame, sequence_length: int = 20, target_col: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert data into sequences for LSTM processing.
    
    Args:
        df: Input DataFrame with features and label
        sequence_length: Length of each sequence (default: 20)
        target_col: Name of the target column (default: 'label')
    
    Returns:
        Tuple of (sequences array, labels array)
        Shape: (samples, timesteps, features)
    """
    print(f"Creating sequences with length {sequence_length}...")
    
    # Separate features and labels
    if target_col in df.columns:
        labels = df[target_col].values
        features = df.drop(columns=[target_col]).values
    else:
        labels = None
        features = df.values
    
    n_features = features.shape[1]
    n_samples = len(features) - sequence_length + 1
    
    sequences = []
    sequence_labels = []
    
    for i in range(n_samples):
        seq = features[i:i + sequence_length]
        sequences.append(seq)
        
        if labels is not None:
            # Use the label of the last time step in the sequence
            sequence_labels.append(labels[i + sequence_length - 1])
    
    sequences = np.array(sequences)
    
    if labels is not None:
        sequence_labels = np.array(sequence_labels)
        print(f"Created {len(sequences)} sequences")
        print(f"Sequence shape: {sequences.shape} (samples, timesteps, features)")
        print(f"Label shape: {sequence_labels.shape}")
        return sequences, sequence_labels
    else:
        print(f"Created {len(sequences)} sequences")
        print(f"Sequence shape: {sequences.shape} (samples, timesteps, features)")
        return sequences, None


def shuffle_and_split(sequences: np.ndarray, labels: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle dataset and split into train and test sets.
    
    Args:
        sequences: Input sequences
        labels: Corresponding labels
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print(f"Shuffling and splitting data (test_size={test_size})...")
    
    # Shuffle data
    indices = np.arange(len(sequences))
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    sequences_shuffled = sequences[indices]
    labels_shuffled = labels[indices]
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        sequences_shuffled,
        labels_shuffled,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_shuffled
    )
    
    print(f"Train set: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test set: {X_test.shape}, Labels: {y_test.shape}")
    print(f"Train label distribution: {np.bincount(y_train)}")
    print(f"Test label distribution: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test


def prepare_dataset(df: pd.DataFrame, sequence_length: int = 20, test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Complete pipeline: create sequences, shuffle, and split.
    
    Args:
        df: Input DataFrame
        sequence_length: Length of each sequence (default: 20)
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary containing train/test splits and metadata
    """
    print("=" * 50)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 50)
    
    # Create sequences
    sequences, labels = create_sequences(df, sequence_length)
    
    # Shuffle and split
    X_train, X_test, y_train, y_test = shuffle_and_split(
        sequences, labels, test_size, random_state
    )
    
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'sequence_length': sequence_length,
        'n_features': sequences.shape[2],
        'n_classes': len(np.unique(labels))
    }
    
    print("\nDataset preparation completed.")
    print(f"Sequence length: {sequence_length}")
    print(f"Number of features: {result['n_features']}")
    print(f"Number of classes: {result['n_classes']}")
    
    return result


def save_preprocessed_data(data: dict, save_path: str = "data/preprocessed_data.pkl"):
    """
    Save preprocessed data to disk.
    
    Args:
        data: Dictionary containing preprocessed data
        save_path: Path to save the data
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(data, save_path)
    print(f"Preprocessed data saved to {save_path}")


def load_preprocessed_data(load_path: str = "data/preprocessed_data.pkl") -> dict:
    """
    Load preprocessed data from disk.
    
    Args:
        load_path: Path to load the data from
    
    Returns:
        Dictionary containing preprocessed data
    """
    data = joblib.load(load_path)
    print(f"Preprocessed data loaded from {load_path}")
    return data


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("..")
    from loader import load_data_efficient
    from preprocessing import preprocess_data
    from vif_filter import remove_high_vif_features
    
    df = load_data_efficient("data/raw")
    df_clean, _ = preprocess_data(df, save_scaler=False)
    df_filtered, _ = remove_high_vif_features(df_clean)
    
    data = prepare_dataset(df_filtered, sequence_length=20)
    save_preprocessed_data(data)
