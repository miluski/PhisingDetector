import pandas as pd
import pickle
import os
import subprocess
import sys
from .url_extractor import URLExtractor
from .url_feature_analyzer import URLFeatureAnalyzer
from .rule_based_detector import RuleBasedDetector

class DataProcessor:
    @staticmethod
    def get_feature_names():
        return URLFeatureAnalyzer.get_feature_names()

    @staticmethod
    def prepare_prediction_data(sender, content, feature_names=None, receiver=None, message_type='email', print_extracted=True):
        if feature_names is None:
            feature_names = DataProcessor.get_feature_names()
            
        if message_type == 'email' and '@' in sender:
            features, analyzed_urls = DataProcessor.analyze_email_with_average_scoring(
                sender, content, receiver, print_extracted
            )
            if print_extracted and analyzed_urls:
                print(f"URLs analyzed: {[f'{source}: {url}' for source, url in analyzed_urls]}")
        else:
            features = DataProcessor.extract_features_from_text(sender, content, receiver, print_extracted, message_type)
            analyzed_urls = []
        
        if features is None:
            return None, analyzed_urls
        
        feature_vector = []
        for feature_name in feature_names:
            feature_vector.append(features.get(feature_name, 0))
        
        return pd.DataFrame([feature_vector], columns=feature_names), analyzed_urls

    @staticmethod
    def extract_features_from_text(sender, content, receiver=None, print_extracted=True, message_type='email'):
        if print_extracted:
            print("Analyzing message content for features")
        
        extracted_url = URLExtractor.extract_url_from_message(sender, content, receiver, message_type, print_extracted)
        
        if not extracted_url:
            return None
        
        features = URLFeatureAnalyzer.analyze_url_features(extracted_url)
        
        if features and print_extracted:
            print(f"Successfully extracted {len(features)} features from URL: {extracted_url}")
        
        return features

    @staticmethod
    def analyze_email_with_average_scoring(sender, content, receiver=None, print_extracted=True):
        url_data = URLExtractor.extract_both_urls_for_email(sender, content, receiver, print_extracted)
        
        features_list = []
        analyzed_urls = []
        
        if url_data['content_url']:
            if print_extracted:
                print(f"Analyzing content URL: {url_data['content_url']}")
            content_features = URLFeatureAnalyzer.analyze_url_features(url_data['content_url'])
            if content_features:
                features_list.append(content_features)
                analyzed_urls.append(('content', url_data['content_url']))

        if url_data['sender_domain_url']:
            if print_extracted:
                print(f"Analyzing sender domain: {url_data['sender_domain_url']}")
            sender_features = URLFeatureAnalyzer.analyze_url_features(url_data['sender_domain_url'])
            if sender_features:
                features_list.append(sender_features)
                analyzed_urls.append(('sender_domain', url_data['sender_domain_url']))
        
        if not features_list:
            if print_extracted:
                print("No valid URLs could be analyzed")
            return None, analyzed_urls
        
        if len(features_list) > 1:
            if print_extracted:
                print(f"Calculating average features from {len(features_list)} URLs")
            averaged_features = {}
            feature_names = DataProcessor.get_feature_names()
            
            for feature_name in feature_names:
                values = [features.get(feature_name, 0) for features in features_list]
                averaged_features[feature_name] = sum(values) / len(values)
            
            return averaged_features, analyzed_urls
        else:
            if print_extracted:
                print(f"Using features from single URL analysis")
            return features_list[0], analyzed_urls

    @staticmethod
    def load_model(model_path):
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                return data['model'], data['scaler'], data['feature_names']
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None, None

    @staticmethod
    def format_prediction_result(prediction, probability, model_type="unknown", analyzed_urls=None):
        result = {
            "is_phishing": bool(prediction),
            "confidence": float(probability) if prediction else float(1 - probability),
            "probability": float(probability),
            "model_type": model_type,
            "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
        }
        
        if analyzed_urls:
            result["analyzed_urls"] = [{"source": source, "url": url} for source, url in analyzed_urls]
            
        return result

    @staticmethod
    def train_model_from_notebook(notebook_path):
        try:
            result = subprocess.run([
                sys.executable, "-m", "jupyter", "nbconvert", 
                "--to", "notebook", "--execute", notebook_path,
                "--inplace"
            ], capture_output=True, text=True, cwd=os.path.dirname(notebook_path))
            
            if result.returncode == 0:
                print(f"Successfully executed notebook: {notebook_path}")
                return True
            else:
                print(f"Error executing notebook: {result.stderr}")
                return False
        except Exception as e:
            print(f"Error training model from notebook: {e}")
            return False
