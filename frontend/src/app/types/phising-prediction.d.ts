export interface PhishingPrediction {
  is_phishing: boolean;
  confidence: number;
  model_type: string;
}