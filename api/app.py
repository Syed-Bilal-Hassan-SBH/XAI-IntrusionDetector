"""
Flask API for XAI-E-DiD
Provides /detect endpoint for anomaly detection
"""

from flask import Flask, request, jsonify
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.inference import InferenceEngine


app = Flask(__name__)

# Initialize inference engine
inference_engine = None


def initialize_engine(model_path: str = "models/model.pt", scaler_path: str = "models/scaler.pkl"):
    """Initialize the inference engine"""
    global inference_engine
    inference_engine = InferenceEngine(model_path, scaler_path)
    print("Inference engine initialized")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': inference_engine is not None
    })


@app.route('/detect', methods=['POST'])
def detect():
    """
    Detect endpoint.
    
    Input: JSON with 'data' field containing 2D array
    Output: Alert JSON with detection results
    """
    if inference_engine is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get data from request
        data = request.json
        
        if 'data' not in data:
            return jsonify({'error': 'Missing "data" field in request'}), 400
        
        # Convert to numpy array
        sample = np.array(data['data'])
        
        # Make prediction
        result = inference_engine.predict_with_explanation(sample)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/batch_detect', methods=['POST'])
def batch_detect():
    """
    Batch detect endpoint.
    
    Input: JSON with 'samples' field containing list of samples
    Output: List of alert JSONs
    """
    if inference_engine is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        if 'samples' not in data:
            return jsonify({'error': 'Missing "samples" field in request'}), 400
        
        samples = data['samples']
        results = inference_engine.batch_predict(samples)
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    """Set detection threshold"""
    if inference_engine is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        if 'threshold' not in data:
            return jsonify({'error': 'Missing "threshold" field in request'}), 400
        
        threshold = float(data['threshold'])
        inference_engine.set_threshold(threshold)
        
        return jsonify({'message': f'Threshold set to {threshold}'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='XAI-E-DiD API Server')
    parser.add_argument('--model', type=str, default='models/model.pt', help='Path to model')
    parser.add_argument('--scaler', type=str, default='models/scaler.pkl', help='Path to scaler')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    # Initialize engine
    initialize_engine(args.model, args.scaler)
    
    # Run app
    print(f"Starting API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
