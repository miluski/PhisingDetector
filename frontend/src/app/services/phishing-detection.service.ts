import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import {
  AnalysisMode,
  DatasetType,
  ModelId,
  ModelAccuracyResponse,
  MessageExamplesResponse,
  PhishingAnalysisResponse,
} from '../models/phishing-analysis.model';

@Injectable({
  providedIn: 'root',
})
export class PhishingDetectionService {
  private static readonly API_URL = 'https://localhost:5001/api';

  constructor(private readonly http: HttpClient) {}

  public getModelAccuracies(): Observable<ModelAccuracyResponse> {
    return this.http.get<ModelAccuracyResponse>(
      `${PhishingDetectionService.API_URL}/model-accuracies`,
    );
  }

  public getMessageExamples(): Observable<MessageExamplesResponse> {
    return this.http.get<MessageExamplesResponse>(
      `${PhishingDetectionService.API_URL}/message-examples`,
    );
  }

  public analyzeEmail(formData: FormData): Observable<PhishingAnalysisResponse> {
    return this.http.post<PhishingAnalysisResponse>(
      `${PhishingDetectionService.API_URL}/analyze-email`, formData,
    );
  }

  public analyzeSms(
    content: string,
    models: readonly ModelId[],
    mode: AnalysisMode,
    dataset: DatasetType
  ): Observable<PhishingAnalysisResponse> {
    const formData = new FormData();
    formData.append('content', content);
    formData.append('mode', mode);
    formData.append('dataset', dataset);
    models.forEach((model) => formData.append('models', model));
    return this.http.post<PhishingAnalysisResponse>(
      `${PhishingDetectionService.API_URL}/analyze-sms`, formData,
    );
  }
}
