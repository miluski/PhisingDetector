export type DatasetType = 'current' | 'url_corpus';

export type AnalysisMode = 'optimized' | 'full';
export type AnalysisType = 'email' | 'sms';

export type ModelId =
  | 'random_forest'
  | 'xgboost'
  | 'mlp'
  | 'svm'
  | 'logistic_regression';

/** Number of input features in each dataset's published feature schema. */
export const DATASET_FEATURE_COUNTS: Record<DatasetType, Record<AnalysisMode, number>> = {
  current: {
    optimized: 74,
    full: 111,
  },
  url_corpus: {
    optimized: 47,
    full: 110,
  },
};

export type ModelTestAccuracy = Record<
  DatasetType,
  Record<AnalysisMode, Partial<Record<ModelId, number>>>
>;

export interface ModelAccuracyResponse {
  success: boolean;
  accuracies: ModelTestAccuracy;
}

export interface EmailExample {
  sender: string;
  recipient: string;
  subject?: string;
  body: string;
  contains_phishing_url: boolean;
}

export interface SmsExample {
  body: string;
  contains_phishing_url: boolean;
}

export interface MessageExamplesResponse {
  success: boolean;
  emails: EmailExample[];
  sms: SmsExample[];
}

export const EMPTY_MODEL_TEST_ACCURACY: ModelTestAccuracy = {
  current: { full: {}, optimized: {} },
  url_corpus: { full: {}, optimized: {} },
};

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
  raw_is_phishing?: boolean;
  receiver_domain?: boolean;
  source?: string;
  classification_note?: string;
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

export interface ModelOption {
  readonly id: ModelId;
  readonly name: string;
  readonly description: string;
}

export interface DatasetOption {
  readonly id: DatasetType;
  readonly name: string;
  readonly description: string;
}

export const AVAILABLE_MODELS: readonly ModelOption[] = [
  { id: 'random_forest', name: 'Random Forest', description: 'Bagging zespołowy' },
  { id: 'xgboost', name: 'XGBoost', description: 'Gradient boosting' },
  { id: 'mlp', name: 'MLP', description: 'Sieć neuronowa' },
  { id: 'svm', name: 'SVM', description: 'Model oparty na jądrze' },
  {
    id: 'logistic_regression',
    name: 'Regresja logistyczna',
    description: 'Model liniowy',
  },
];

export const AVAILABLE_DATASETS: readonly DatasetOption[] = [
  {
    id: 'current',
    name: 'Zbiór bazowy (Grabarica)',
    description: 'Oryginalny zbiór danych Grabarica',
  },
  {
    id: 'url_corpus',
    name: 'Rozszerzony zbiór adresów URL',
    description: 'Phishingowe feedy oraz bezpieczne domeny z rankingu Tranco',
  },
];
