"""
Flask API server for XAI-E-DiD
"""

from flask import Flask, request, jsonify
import torch
import numpy as np
from pathlib import Path

from .config import Config
from .model import ModelFactory


class APIServer:
    """Flask API server for model inference"""
    
    def __init__(self, config: Config):
        self.config = config
        self.app = Flask(__name__)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        # Load model
        self._load_model()
        
        # Setup routes
        self._setup_routes()
    
    def _load_model(self):
        """Load trained model"""
        model_dir = Path(self.config.get('paths.model_save_dir', 'models'))
        model_path = model_dir / "best_model.pth"
        
        if not model_path.exists():
            print(f"Warning: No model found at {model_path}")
            return
        
        self.model = ModelFactory.create_model(self.config)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {model_path}")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint"""
            return jsonify({'status': 'healthy', 'model_loaded': self.model is not None})
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Prediction endpoint"""
            if self.model is None:
                return jsonify({'error': 'Model not loaded'}), 500
            
            try:
                data = request.json
                features = np.array(data['features'])
                
                # Ensure correct shape
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                
                # Convert to tensor
                features_tensor = torch.FloatTensor(features).to(self.device)
                
                # Predict
                with torch.no_grad():
                    outputs = self.model(features_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                
                return jsonify({
                    'predictions': predictions.cpu().tolist(),
                    'probabilities': probabilities.cpu().tolist()
                })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/batch_predict', methods=['POST'])
        def batch_predict():
            """Batch prediction endpoint"""
            if self.model is None:
                return jsonify({'error': 'Model not loaded'}), 500
            
            try:
                data = request.json
                features = np.array(data['features'])
                
                # Convert to tensor
                features_tensor = torch.FloatTensor(features).to(self.device)
                
                # Predict
                with torch.no_grad():
                    outputs = self.model(features_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                
                return jsonify({
                    'predictions': predictions.cpu().tolist(),
                    'probabilities': probabilities.cpu().tolist(),
                    'count': len(predictions)
                })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 400
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the API server"""
        print(f"Starting API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
