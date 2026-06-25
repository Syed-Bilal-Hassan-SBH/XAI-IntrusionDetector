"""
VIF (Variance Inflation Factor) feature filter for XAI-E-DiD
Removes multicollinear features
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Tuple
import warnings

warnings.filterwarnings('ignore')


def calculate_vif(df: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """
    Calculate VIF for all features in the DataFrame.
    
    Args:
        df: Input DataFrame (should not contain label column)
        threshold: VIF threshold for feature removal
    
    Returns:
        DataFrame with VIF values for each feature
    """
    print(f"Calculating VIF for {df.shape[1]} features...")
    
    # Ensure all columns are numeric
    df = df.select_dtypes(include=[np.number])
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    
    vif_data = vif_data.sort_values("VIF", ascending=False)
    
    print("\nVIF Values:")
    print(vif_data.to_string(index=False))
    
    return vif_data


def remove_high_vif_features(df: pd.DataFrame, threshold: float = 10.0, max_iterations: int = 10) -> Tuple[pd.DataFrame, List[str]]:
    """
    Iteratively remove features with VIF > threshold.
    
    Args:
        df: Input DataFrame
        threshold: VIF threshold for removal
        max_iterations: Maximum iterations to prevent infinite loops
    
    Returns:
        Tuple of (DataFrame with reduced features, list of removed features)
    """
    print(f"\nRemoving features with VIF > {threshold}...")
    
    # Separate label column if exists
    label_col = None
    if 'label' in df.columns:
        label_col = df['label']
        df = df.drop(columns=['label'])
    
    removed_features = []
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Calculate VIF
        vif_data = calculate_vif(df, threshold)
        
        # Find features with VIF > threshold
        high_vif_features = vif_data[vif_data["VIF"] > threshold]["Feature"].tolist()
        
        if not high_vif_features:
            print(f"No features with VIF > {threshold} found.")
            break
        
        # Remove feature with highest VIF
        feature_to_remove = high_vif_features[0]
        removed_features.append(feature_to_remove)
        df = df.drop(columns=[feature_to_remove])
        
        print(f"Iteration {iteration}: Removed '{feature_to_remove}' (VIF: {vif_data.iloc[0]['VIF']:.2f})")
        print(f"Remaining features: {df.shape[1]}")
    
    # Add label back if it was removed
    if label_col is not None:
        df['label'] = label_col
    
    print(f"\nVIF filtering completed.")
    print(f"Removed {len(removed_features)} features")
    print(f"Final shape: {df.shape}")
    
    return df, removed_features


def apply_vif_filter(df: pd.DataFrame, features_to_keep: List[str]) -> pd.DataFrame:
    """
    Apply VIF filter to new data by keeping only specified features.
    
    Args:
        df: Input DataFrame
        features_to_keep: List of features to keep
    
    Returns:
        Filtered DataFrame
    """
    print(f"Applying VIF filter - keeping {len(features_to_keep)} features...")
    
    # Separate label column if exists
    label_col = None
    if 'label' in df.columns:
        label_col = df['label']
    
    # Keep only specified features
    df_filtered = df[features_to_keep].copy()
    
    # Add label back if it was removed
    if label_col is not None:
        df_filtered['label'] = label_col
    
    print(f"Filtered shape: {df_filtered.shape}")
    
    return df_filtered


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("..")
    from loader import load_data_efficient
    from preprocessing import preprocess_data
    
    df = load_data_efficient("data/raw")
    df_clean, _ = preprocess_data(df, save_scaler=False)
    df_filtered, removed = remove_high_vif_features(df_clean)
    print(f"\nRemoved features: {removed}")
