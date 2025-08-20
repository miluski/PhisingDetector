import { PhishingPrediction } from './phising-prediction';

export interface EnsembleResponse {
  timestamp: string;
  input: {
    sender: string;
    receiver: string;
    content_length: number;
    sender_type: string;
  };
  individual_predictions: {
    [key: string]: PhishingPrediction;
  };
  ensemble_prediction: {
    is_phishing: boolean;
    confidence: number;
    risk_level: string;
    algorithms_agreeing: number;
    total_algorithms: number;
  };
  plots: {
    logistic_regression: string;
    random_forest: string;
    xgboost: string;
    svm: string;
    mlp: string;
    comparison: string;
  };
}