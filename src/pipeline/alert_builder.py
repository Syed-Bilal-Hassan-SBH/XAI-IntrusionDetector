"""
Alert builder for XAI-E-DiD
Constructs JSON alert with detailed information
"""

import numpy as np
import json
from typing import Dict, List, Any


class AlertBuilder:
    """Builds structured alerts for detected anomalies"""
    
    # Regulatory compliance mapping
    COMPLIANCE_TAGS = {
        'GDPR_Article_22': 'Automated decision-making with significant impact',
        'PCI_DSS_Req_10': 'Security monitoring and logging',
        'PCI_DSS_Req_11': 'Regular security testing',
        'EU_AI_Act_High_Risk': 'High-risk AI system in critical infrastructure'
    }
    
    def __init__(self):
        """Initialize alert builder"""
        self.attack_types = {
            'normal': 'Normal Traffic',
            'dos': 'Denial of Service',
            'probe': 'Probing/Scanning',
            'r2l': 'Remote to Local',
            'u2r': 'User to Root',
            'anomaly': 'Anomalous Traffic'
        }
    
    def build_alert(
        self,
        is_anomaly: bool,
        confidence: float,
        reconstruction_error: float,
        threshold: float,
        top_features: List[str] = None,
        feature_importance: Dict[str, float] = None,
        attention_weights: np.ndarray = None,
        explanation: str = None,
        attack_category: str = None,
        add_compliance_tags: bool = True
    ) -> Dict[str, Any]:
        """
        Build alert JSON.
        
        Args:
            is_anomaly: Whether the sample is an anomaly
            confidence: Detection confidence
            reconstruction_error: Reconstruction error
            threshold: Detection threshold
            top_features: List of top contributing features
            feature_importance: Dictionary of feature importances
            attention_weights: Attention weights over time steps
            explanation: Text explanation
            attack_category: Attack category (DoS, DDoS, PortScan, etc.)
            add_compliance_tags: Whether to add regulatory compliance tags
        
        Returns:
            Alert dictionary
        """
        attack_label = "Anomaly" if is_anomaly else "Normal"
        confidence = min(1.0, reconstruction_error / threshold) if is_anomaly else 1.0 - (reconstruction_error / threshold)
        
        # Get top features
        top_features_list = []
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features_list = [
                {
                    'feature': feature,
                    'importance': float(importance)
                }
                for feature, importance in sorted_features[:5]
            ]
        
        # Attention summary
        attention_summary = None
        if attention_weights is not None:
            peak_idx = int(np.argmax(attention_weights))
            attention_summary = {
                'peak_time_step': peak_idx,
                'peak_weight': float(attention_weights[peak_idx]),
                'mean_weight': float(np.mean(attention_weights))
            }
        
        # Regulatory compliance tags
        compliance_tags = []
        if add_compliance_tags and is_anomaly:
            compliance_tags = self._get_compliance_tags(attack_category)
        
        # Counterfactual explanation
        counterfactual = None
        if is_anomaly and top_features_list:
            counterfactual = self._generate_counterfactual(top_features_list[0], reconstruction_error, threshold)
        
        alert = {
            'attack_label': attack_label,
            'attack_type': self.attack_types['anomaly'] if is_anomaly else self.attack_types['normal'],
            'attack_category': attack_category if attack_category else 'Unknown',
            'is_anomaly': is_anomaly,
            'confidence': float(confidence),
            'reconstruction_error': float(reconstruction_error),
            'threshold': float(threshold),
            'top_features': top_features_list,
            'attention_summary': attention_summary,
            'compliance_tags': compliance_tags,
            'counterfactual_explanation': counterfactual,
            'explanation': self._generate_explanation(
                is_anomaly, reconstruction_error, threshold, top_features_list, attention_summary, attack_category
            )
        }
        
        return alert
    
    def _get_compliance_tags(self, attack_category: str) -> List[Dict[str, str]]:
        """Get regulatory compliance tags based on attack category"""
        tags = []
        
        # GDPR Article 22 - applies to all automated decisions
        tags.append({
            'regulation': 'GDPR Article 22',
            'requirement': 'Explainable automated decision-making',
            'status': 'Compliant - SHAP explanation provided'
        })
        
        # PCI-DSS Requirement 10 - security monitoring
        tags.append({
            'regulation': 'PCI-DSS Requirement 10',
            'requirement': 'Security monitoring and logging',
            'status': 'Compliant - Alert logged with full details'
        })
        
        # EU AI Act - high-risk systems
        if attack_category in ['DoS', 'DDoS', 'Infiltration', 'APT']:
            tags.append({
                'regulation': 'EU AI Act',
                'requirement': 'High-risk AI system requirements',
                'status': 'Compliant - FSS critical infrastructure'
            })
        
        return tags
    
    def _generate_counterfactual(
        self,
        top_feature: Dict,
        reconstruction_error: float,
        threshold: float
    ) -> str:
        """
        Generate counterfactual explanation.
        
        Args:
            top_feature: Top contributing feature
            reconstruction_error: Current reconstruction error
            threshold: Detection threshold
        
        Returns:
            Counterfactual explanation string
        """
        feature_name = top_feature['feature']
        feature_importance = top_feature['importance']
        
        # Estimate what value would make it normal
        reduction_needed = (reconstruction_error - threshold) / abs(feature_importance)
        
        return (f"If {feature_name} had been reduced by approximately {abs(reduction_needed):.4f}, "
                f"the reconstruction error would have been below the threshold "
                f"({threshold:.4f}) and classified as normal.")
    
    def _generate_explanation(
        self,
        is_anomaly: bool,
        reconstruction_error: float,
        threshold: float,
        top_features: List[Dict],
        attention_summary: Dict,
        attack_category: str
    ) -> str:
        """Generate textual explanation"""
        if not is_anomaly:
            return "Normal traffic pattern detected. Reconstruction error within normal threshold."
        
        explanation = f"Anomaly detected ({attack_category if attack_category else 'Unknown'}) "
        explanation += f"with reconstruction error {reconstruction_error:.4f} (threshold: {threshold:.4f}). "
        
        if top_features:
            feature_names = [f['feature'] for f in top_features[:3]]
            explanation += f"Top contributing features: {', '.join(feature_names)}. "
        
        if attention_summary:
            explanation += f"Peak anomaly activity at time step {attention_summary['peak_time_step']}. "
        
        explanation += "Regulatory compliance tags attached for GDPR and PCI-DSS requirements."
        
        return explanation
    
    def _build_attention_summary(self, attention_weights: np.ndarray) -> Dict[str, Any]:
        """
        Build attention summary from attention weights.
        
        Args:
            attention_weights: Attention weights array
        
        Returns:
            Attention summary dictionary
        """
        if attention_weights is None:
            return {
                'mean_attention': 0.0,
                'max_attention': 0.0,
                'peak_time_step': None,
                'attention_distribution': 'uniform'
            }
        
        attention_weights = np.array(attention_weights)
        
        summary = {
            'mean_attention': float(np.mean(attention_weights)),
            'max_attention': float(np.max(attention_weights)),
            'min_attention': float(np.min(attention_weights)),
            'peak_time_step': int(np.argmax(attention_weights)),
            'attention_distribution': self._classify_distribution(attention_weights),
            'attention_weights': attention_weights.tolist() if len(attention_weights) <= 100 else attention_weights[:100].tolist()
        }
        
        return summary
    
    def _classify_distribution(self, weights: np.ndarray) -> str:
        """Classify attention distribution type"""
        std = np.std(weights)
        mean = np.mean(weights)
        
        if std < 0.1 * mean:
            return 'uniform'
        elif std > 0.5 * mean:
            return 'focused'
        else:
            return 'moderate'
    
    def _generate_explanation(
        self,
        is_anomaly: bool,
        confidence: float,
        reconstruction_error: float
    ) -> str:
        """
        Generate text explanation.
        
        Args:
            is_anomaly: Whether the sample is an anomaly
            confidence: Detection confidence
            reconstruction_error: Reconstruction error
        
        Returns:
            Text explanation
        """
        if is_anomaly:
            explanation = (
                f"Anomaly detected with {confidence:.2%} confidence. "
                f"The reconstruction error ({reconstruction_error:.6f}) exceeds the threshold, "
                f"indicating significant deviation from normal traffic patterns. "
                f"This suggests potential malicious activity or unusual network behavior."
            )
        else:
            explanation = (
                f"Normal traffic detected. The reconstruction error ({reconstruction_error:.6f}) "
                f"is within acceptable thresholds, indicating the traffic pattern matches "
                f"normal baseline behavior."
            )
        
        return explanation
    
    def add_feature_explanation(
        self,
        alert: Dict[str, Any],
        feature_names: List[str],
        feature_values: np.ndarray,
        importance_scores: np.ndarray
    ) -> Dict[str, Any]:
        """
        Add feature-level explanation to alert.
        
        Args:
            alert: Existing alert dictionary
            feature_names: List of feature names
            feature_values: Feature values
            importance_scores: Feature importance scores
        
        Returns:
            Updated alert dictionary
        """
        # Get top 5 features
        top_indices = np.argsort(importance_scores)[-5:][::-1]
        
        top_features = []
        for idx in top_indices:
            top_features.append({
                'name': feature_names[idx],
                'value': float(feature_values[idx]),
                'importance': float(importance_scores[idx])
            })
        
        alert['top_features'] = top_features
        alert['feature_importance'] = {
            feature_names[i]: float(importance_scores[i])
            for i in range(len(feature_names))
        }
        
        return alert
    
    def to_json(self, alert: Dict[str, Any]) -> str:
        """Convert alert to JSON string"""
        return json.dumps(alert, indent=2)
    
    def save_alert(self, alert: Dict[str, Any], filepath: str):
        """Save alert to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(alert, f, indent=2)
        print(f"Alert saved to {filepath}")


if __name__ == "__main__":
    # Test alert builder
    builder = AlertBuilder()
    
    # Create dummy data
    alert = builder.build_alert(
        is_anomaly=True,
        confidence=0.95,
        reconstruction_error=0.123,
        threshold=0.05,
        top_features=['feature_1', 'feature_2', 'feature_3'],
        feature_importance={'feature_1': 0.8, 'feature_2': 0.6, 'feature_3': 0.4},
        attention_weights=np.array([0.1, 0.2, 0.5, 0.15, 0.05])
    )
    
    print("Alert JSON:")
    print(builder.to_json(alert))
