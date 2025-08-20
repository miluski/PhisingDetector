import pandas as pd
import pickle
import os
from .url_feature_analyzer import URLFeatureAnalyzer

class ModelAnalyzer:
    @staticmethod
    def get_model_statistics(model_path):
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            stats = {
                'model_type': str(type(data.get('model', 'Unknown'))),
                'scaler_type': str(type(data.get('scaler', 'None'))),
                'feature_count': len(data.get('feature_names', [])),
                'metrics': data.get('metrics', {}),
                'best_params': data.get('best_params', {})
            }
            
            if 'scaler' in data and hasattr(data['scaler'], 'mean_'):
                scaler = data['scaler']
                feature_names = data.get('feature_names', [])
                scaling_issues = []
                
                for i, (name, mean, scale) in enumerate(zip(feature_names, scaler.mean_, scaler.scale_)):
                    if scale < 0.001 or scale > 100:
                        scaling_issues.append({'feature': name, 'mean': mean, 'scale': scale})
                
                stats['scaling_issues'] = scaling_issues
            
            return stats
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def get_feature_distribution_comparison():
        try:
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'dataset_full.csv')
            df = pd.read_csv(data_path)
            
            distribution_stats = {
                'total_samples': len(df),
                'feature_count': len([col for col in df.columns if col not in ['FILENAME', 'URL', 'Domain', 'TLD', 'Title', 'label']]),
                'class_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else {}
            }
            
            if 'URLCharProb' in df.columns:
                distribution_stats['URLCharProb'] = {
                    'min': df['URLCharProb'].min(),
                    'max': df['URLCharProb'].max(),
                    'mean': df['URLCharProb'].mean(),
                    'std': df['URLCharProb'].std()
                }
            
            return distribution_stats
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def get_scaling_analysis(url):
        try:
            features = URLFeatureAnalyzer.analyze_url_features(url)
            if not features:
                return {'error': 'Failed to extract features'}
            
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            
            scaling_results = {}
            
            for model_file in model_files:
                model_path = os.path.join(models_dir, model_file)
                try:
                    with open(model_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    if 'scaler' in data and hasattr(data['scaler'], 'transform'):
                        scaler = data['scaler']
                        feature_names = data.get('feature_names', [])
                        
                        feature_vector = [features.get(name, 0) for name in feature_names]
                        scaled_features = scaler.transform([feature_vector])[0]
                        
                        scaling_results[model_file] = {
                            'original_features': dict(zip(feature_names, feature_vector)),
                            'scaled_features': dict(zip(feature_names, scaled_features)),
                            'extreme_values': [name for name, val in zip(feature_names, scaled_features) if abs(val) > 10]
                        }
                except Exception as e:
                    scaling_results[model_file] = {'error': str(e)}
            
            return scaling_results
        except Exception as e:
            return {'error': str(e)}
