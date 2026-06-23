export type DatasetType = 'current' | 'url_corpus';

export interface PhishingAnalysisResponse {
  success: boolean;
  is_phishing: boolean;
  dataset?: DatasetType;
  message?: string;
  error?: string;
  urls_found: number;
  urls: string[];
  analysis: UrlAnalysis[];
  container_id?: string;
}

export interface UrlAnalysis {
  url: string;
  success: boolean;
  error?: string;
  predictions?: ModelPrediction[];
  ensemble?: EnsemblePrediction;
  features?: Record<string, number>;
  visualizations?: Visualizations;
  similar_samples?: SimilarSample[];
  is_phishing: boolean;
}

export interface SimilarSample {
  index: number;
  distance: number;
  is_phishing: boolean;
  features: Record<string, number>;
}

export interface ModelPrediction {
  model: string;
  mode: string;
  dataset?: DatasetType;
  prediction: number;
  prediction_label: string;
  probabilities: {
    safe: number;
    phishing: number;
  };
  confidence: number;
  features_used: number;
  voting?: TreeVoting;
  top_features?: Feature[];
}

export interface TreeVoting {
  safe: number;
  phishing: number;
  total_trees: number;
  safe_percentage: number;
  phishing_percentage: number;
}

export interface Feature {
  name: string;
  importance: number;
  value: number;
}

export interface EnsemblePrediction {
  ensemble_prediction: number;
  ensemble_label: string;
  dataset?: DatasetType;
  models_voting: {
    phishing: number;
    safe: number;
    total: number;
  };
  average_phishing_probability: number;
  confidence: number;
  individual_predictions: ModelPrediction[];
}

export interface Visualizations {
  url: string;
  charts: {
    probabilities?: string;
    voting?: string;
    feature_importance?: string;
    ensemble_voting?: string;
  };
}

export const AVAILABLE_MODELS = [
  { id: 'random_forest', name: 'Random Forest', description: 'Bagging zespołowy (96% dokładność)' },
  { id: 'xgboost', name: 'XGBoost', description: 'Gradient boosting (97% dokładność)' },
  { id: 'mlp', name: 'MLP', description: 'Sieć neuronowa (95% dokładność)' },
  { id: 'svm', name: 'SVM', description: 'Oparty na jądrze (94% dokładność)' },
  {
    id: 'logistic_regression',
    name: 'Regresja logistyczna',
    description: 'Model liniowy (92% dokładność)',
  },
];

export const AVAILABLE_DATASETS = [
  { id: 'current', name: 'Obecny zbiór', description: 'Aktualny zbiór bazowy projektu' },
  { id: 'url_corpus', name: 'URL corpus', description: 'Nowy zbiór z feedów phishing + Tranco' },
];
