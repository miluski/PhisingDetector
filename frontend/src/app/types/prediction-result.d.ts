export interface PredictionResult {
  predictions: { [key: string]: number };
  confidences: { [key: string]: number };
  ensemble_prediction?: number;
  ensemble_confidence?: number;
}