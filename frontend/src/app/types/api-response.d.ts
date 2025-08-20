import { PhishingPrediction } from './phising-prediction';

export interface ApiResponse {
  algorithm: string;
  prediction: PhishingPrediction;
  input: {
    sender: string;
    receiver: string;
    content_length: number;
    sender_type?: string;
  };
  plots?: {
    analysis: string;
    comparison: string;
  };
}