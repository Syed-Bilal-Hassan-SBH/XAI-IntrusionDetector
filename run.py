#!/usr/bin/env python3
"""
XAI-E-DiD - Explainable AI for Early Detection of Intrusion Detection
Main entry point for the project.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="XAI-E-DiD: Explainable AI for Early Detection of Intrusion Detection"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model"
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Run detection on new data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Directory containing data files"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save/load models"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=20,
        help="Sequence length for LSTM"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for LSTM"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    # Ensure at least one mode is selected
    if not any([args.train, args.evaluate, args.detect]):
        parser.print_help()
        print("\nError: Please specify at least one mode (--train, --evaluate, or --detect)")
        return 1
    
    # Training mode
    if args.train:
        print("=" * 60)
        print("TRAINING MODE")
        print("=" * 60)
        
        from training.train import train_model
        
        try:
            train_model(
                data_dir=args.data_dir,
                model_save_dir=args.model_dir,
                sequence_length=args.seq_len,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr
            )
            print("Training completed successfully.")
        except Exception as e:
            print(f"Training failed: {e}")
            return 1
    
    # Evaluation mode
    if args.evaluate:
        print("\n" + "=" * 60)
        print("EVALUATION MODE")
        print("=" * 60)
        
        from data.loader import load_data_efficient
        from data.preprocessing import preprocess_data
        from data.vif_filter import remove_high_vif_features, apply_vif_filter
        from data.feature_engineering import prepare_dataset
        from evaluation.evaluator import Evaluator
        import joblib
        
        try:
            # Load test data
            print("Loading test data...")
            df = load_data_efficient(args.data_dir)
            
            # Preprocess
            print("Preprocessing data...")
            df_clean, _ = preprocess_data(df, save_scaler=False)
            
            # Load feature info
            feature_info_path = Path(args.model_dir) / "feature_info.pkl"
            if feature_info_path.exists():
                feature_info = joblib.load(feature_info_path)
                df_filtered = apply_vif_filter(df_clean, feature_info['remaining_features'])
            else:
                df_filtered, _ = remove_high_vif_features(df_clean)
            
            # Create sequences
            print("Creating sequences...")
            data_dict = prepare_dataset(df_filtered, sequence_length=args.seq_len)
            
            # Evaluate
            model_path = Path(args.model_dir) / "model.pt"
            scaler_path = Path(args.model_dir) / "scaler.pkl"
            
            evaluator = Evaluator(str(model_path), str(scaler_path), args.results_dir)
            results = evaluator.evaluate(data_dict['X_test'], data_dict['y_test'])
            
            print("Evaluation completed successfully.")
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return 1
    
    # Detection mode
    if args.detect:
        print("\n" + "=" * 60)
        print("DETECTION MODE")
        print("=" * 60)
        
        from pipeline.inference import InferenceEngine
        import numpy as np
        import json
        
        try:
            model_path = Path(args.model_dir) / "model.pt"
            scaler_path = Path(args.model_dir) / "scaler.pkl"
            
            # Initialize inference engine
            engine = InferenceEngine(str(model_path), str(scaler_path))
            
            # Generate sample for demonstration
            print("Generating sample for detection...")
            metadata_path = Path(args.model_dir) / "model_metadata.pkl"
            if metadata_path.exists():
                metadata = joblib.load(metadata_path)
                n_features = metadata['input_dim']
                seq_len = metadata['seq_len']
            else:
                n_features = 784
                seq_len = 20
            
            sample = np.random.randn(seq_len, n_features)
            
            # Make prediction
            result = engine.predict_with_explanation(sample)
            
            print("\nDetection Result:")
            print(json.dumps(result, indent=2))
            
            print("\nDetection completed successfully.")
        except Exception as e:
            print(f"Detection failed: {e}")
            return 1
    
    print("\n" + "=" * 60)
    print("ALL OPERATIONS COMPLETED")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
