from .url_extractor import URLExtractor
from .url_feature_analyzer import URLFeatureAnalyzer
from .rule_based_detector import RuleBasedDetector
from .data_processor import DataProcessor
from .model_analyzer import ModelAnalyzer

class PhishingUtils:
    extract_url_from_message = URLExtractor.extract_url_from_message
    extract_both_urls_for_email = URLExtractor.extract_both_urls_for_email
    
    analyze_url_features = URLFeatureAnalyzer.analyze_url_features
    calculate_similarity_index = URLFeatureAnalyzer.calculate_similarity_index
    calculate_char_continuation_rate = URLFeatureAnalyzer.calculate_char_continuation_rate
    calculate_url_char_probability = URLFeatureAnalyzer.calculate_url_char_probability
    
    rule_based_phishing_check = RuleBasedDetector.rule_based_phishing_check
    
    get_feature_names = DataProcessor.get_feature_names
    prepare_prediction_data = DataProcessor.prepare_prediction_data
    extract_features_from_text = DataProcessor.extract_features_from_text
    analyze_email_with_average_scoring = DataProcessor.analyze_email_with_average_scoring
    load_model = DataProcessor.load_model
    format_prediction_result = DataProcessor.format_prediction_result
    train_model_from_notebook = DataProcessor.train_model_from_notebook
    
    get_model_statistics = ModelAnalyzer.get_model_statistics
    get_feature_distribution_comparison = ModelAnalyzer.get_feature_distribution_comparison
    get_scaling_analysis = ModelAnalyzer.get_scaling_analysis