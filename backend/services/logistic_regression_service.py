import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.phishing_utils import PhishingUtils

class LogisticRegressionService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False

        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(backend_dir, 'models', 'logistic_regression_model.pkl')
        self.notebook_path = os.path.join(backend_dir, 'notebooks', '1_logistic_regression_phishing_detection.ipynb')
    
    def train_model(self):
        if not os.path.exists(self.model_path):
            print(f"Model not found. Training from notebook: {self.notebook_path}")
            if PhishingUtils.train_model_from_notebook(self.notebook_path):
                return self.load_model()
            else:
                print("Failed to train model from notebook")
                return False
        else:
            return self.load_model()
    
    def predict(self, sender, content, receiver=None, message_type='email'):
        if not self.is_trained:
            if not self.train_model():
                return {"error": "Model not trained and training failed"}
        
        try:
            data_result = PhishingUtils.prepare_prediction_data(sender, content, self.feature_names, receiver, message_type)
            if data_result is None:
                return {"error": "Failed to prepare prediction data"}
            
            feature_vector, analyzed_urls = data_result
            if feature_vector is None:
                if message_type == 'sms':
                    return PhishingUtils.format_prediction_result(0, 0.1, "logistic_regression")
                elif '@' not in sender:  
                    return PhishingUtils.format_prediction_result(0, 0.1, "logistic_regression")
                else:
                    return {"error": "No URL found in email content and unable to analyze sender domain"}
            
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            prediction = self.model.predict(feature_vector_scaled)[0]
            probability = self.model.predict_proba(feature_vector_scaled)[0][1]
            
            return PhishingUtils.format_prediction_result(prediction, probability, "logistic_regression", analyzed_urls)
            
        except Exception as e:
            return {"error": f"Prediction error: {e}"}
    
    def predict_enhanced(self, sender, content, receiver=None, message_type='email'):
        """Enhanced prediction using ML model + rule-based detection"""
        return PhishingUtils.enhanced_prediction_with_rules(sender, content, self, message_type)
    
    def load_model(self):
        self.model, self.scaler, self.feature_names = PhishingUtils.load_model(self.model_path)
        if self.model is not None:
            self.is_trained = True
            return True
        return False
