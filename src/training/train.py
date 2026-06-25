"""
Training script for XAI-E-DiD
Complete training pipeline
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib
from pathlib import Path
import argparse
import numpy as np

from data.loader import load_data_efficient
from data.preprocessing import preprocess_data
from data.vif_filter import remove_high_vif_features
from data.feature_engineering import prepare_dataset, save_preprocessed_data
from models.gan_lstm_ae import GANLSTMAutoencoder
from training.trainer import Trainer


def train_model(
    data_dir: str = "data/raw",
    model_save_dir: str = "models",
    sequence_length: int = 20,
    hidden_dim: int = 128,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    use_smote: bool = True,
    smote_sampling_strategy: str = 'auto'
):
    """
    Complete training pipeline.
    
    Args:
        data_dir: Directory containing CSV files
        model_save_dir: Directory to save models
        sequence_length: Length of sequences for LSTM
        hidden_dim: Hidden dimension for LSTM
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    print("=" * 60)
    print("XAI-E-DiD TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/7] Loading data...")
    df = load_data_efficient(data_dir)
    
    # Step 2: Preprocess
    print("\n[2/7] Preprocessing data...")
    df_clean, scaler = preprocess_data(df, save_scaler=True, scaler_path=f"{model_save_dir}/scaler.pkl")
    
    # Step 3: VIF filtering
    print("\n[3/7] Applying VIF filter...")
    df_filtered, removed_features = remove_high_vif_features(df_clean)
    
    # Save removed features for later use
    feature_info = {
        'removed_features': removed_features,
        'remaining_features': df_filtered.drop(columns=['label']).columns.tolist()
    }
    joblib.dump(feature_info, f"{model_save_dir}/feature_info.pkl")
    
    # Step 4: Feature engineering (sequences)
    print("\n[4/7] Creating sequences...")
    data_dict = prepare_dataset(df_filtered, sequence_length=sequence_length)
    
    print(f"Training samples: {data_dict['X_train'].shape[0]}")
    print(f"Test samples: {data_dict['X_test'].shape[0]}")
    
    # Apply SMOTE for class imbalance if enabled
    if use_smote and 'y_train' in data_dict:
        print("Applying SMOTE for class imbalance handling...")
        try:
            from imblearn.over_sampling import SMOTE
            
            # Flatten sequences for SMOTE
            n_samples, seq_len, n_features = data_dict['X_train'].shape
            X_train_flat = data_dict['X_train'].reshape(n_samples, -1)
            
            # Apply SMOTE
            smote = SMOTE(sampling_strategy=smote_sampling_strategy, random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, data_dict['y_train'])
            
            # Reshape back to sequences
            data_dict['X_train'] = X_train_resampled.reshape(-1, seq_len, n_features)
            data_dict['y_train'] = y_train_resampled
            
            print(f"Training samples after SMOTE: {data_dict['X_train'].shape[0]}")
            print(f"Class distribution after SMOTE: {np.bincount(data_dict['y_train'])}")
        except ImportError:
            print("imbalanced-learn not installed, skipping SMOTE")
        except Exception as e:
            print(f"SMOTE failed: {e}, continuing without SMOTE")
    
    # Step 5: Create data loaders
    print("\n[5/7] Creating data loaders...")
    train_dataset = TensorDataset(
        torch.FloatTensor(data_dict['X_train']),
        torch.FloatTensor(data_dict['y_train']).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(data_dict['X_test']),
        torch.FloatTensor(data_dict['y_test']).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Step 6: Initialize model
    print("\n[6/7] Initializing model...")
    model = GANLSTMAutoencoder(
        input_dim=data_dict['n_features'],
        hidden_dim=hidden_dim,
        seq_len=sequence_length,
        num_layers=2
    )
    
    print(f"Model architecture:")
    print(f"  Input dimension: {data_dict['n_features']}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Step 7: Train
    print("\n[7/7] Training model...")
    trainer = Trainer(
        model,
        learning_rate=learning_rate,
        save_dir=model_save_dir
    )
    
    trainer.train(
        train_loader,
        test_loader,
        epochs=epochs,
        early_stopping_patience=10,
        save_best=True
    )
    
    # Save model metadata
    metadata = {
        'input_dim': data_dict['n_features'],
        'hidden_dim': hidden_dim,
        'seq_len': sequence_length,
        'sequence_length': sequence_length,
        'n_classes': data_dict['n_classes']
    }
    joblib.dump(metadata, f"{model_save_dir}/model_metadata.pkl")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Model saved to: {model_save_dir}/model.pt")
    print(f"Scaler saved to: {model_save_dir}/scaler.pkl")
    print(f"Feature info saved to: {model_save_dir}/feature_info.pkl")
    print(f"Model metadata saved to: {model_save_dir}/model_metadata.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XAI-E-DiD model")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="models", help="Model save directory")
    parser.add_argument("--seq_len", type=int, default=20, help="Sequence length")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        model_save_dir=args.model_dir,
        sequence_length=args.seq_len,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
