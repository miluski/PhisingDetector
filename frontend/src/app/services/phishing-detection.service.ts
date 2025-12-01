import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { PhishingAnalysisResponse } from '../models/phishing-analysis.model';

@Injectable({
  providedIn: 'root',
})
export class PhishingDetectionService {
  private apiUrl = 'https://localhost:5001/api';

  constructor(private http: HttpClient) {}

  public analyzeEmail(formData: FormData): Observable<PhishingAnalysisResponse> {
    return this.http.post<PhishingAnalysisResponse>(`${this.apiUrl}/analyze-email`, formData);
  }

  public analyzeSms(
    content: string,
    models: string[],
    mode: 'optimized' | 'full'
  ): Observable<PhishingAnalysisResponse> {
    const formData = new FormData();
    formData.append('content', content);
    formData.append('mode', mode);
    models.forEach((model) => formData.append('models', model));
    return this.http.post<PhishingAnalysisResponse>(`${this.apiUrl}/analyze-sms`, formData);
  }
}
