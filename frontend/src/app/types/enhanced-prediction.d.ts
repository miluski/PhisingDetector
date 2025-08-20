export interface EnhancedPrediction {
  is_phishing: boolean;
  confidence: number;
  probability: number;
  model_type: string;
  risk_level: string;
  ml_prediction: {
    is_phishing: boolean;
    probability: number;
    confidence: number;
  };
  rule_based: {
    is_phishing: boolean;
    probability: number;
    risk_factors: string[];
  };
  analyzed_urls?: Array<{
    source: string;
    url: string;
  }>;
}

export interface ModelStatistics {
  model_type: string;
  scaler_type: string;
  feature_count: number;
  metrics?: any;
  best_params?: any;
  scaling_issues?: Array<{
    feature: string;
    mean: number;
    scale: number;
  }>;
}

export interface FeatureStatistics {
  total_samples: number;
  feature_count: number;
  class_distribution: { [key: string]: number };
  URLCharProb?: {
    min: number;
    max: number;
    mean: number;
    std: number;
  };
}

export interface ScalingStatistics {
  [modelName: string]: {
    original_features: { [feature: string]: number };
    scaled_features: { [feature: string]: number };
    extreme_values: string[];
  } | { error: string };
}
