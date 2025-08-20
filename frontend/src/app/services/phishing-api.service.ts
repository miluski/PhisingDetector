import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import { ApiResponse } from '../types/api-response';
import { EnsembleResponse } from '../types/ensemble-response';
import { ModelPerformance } from '../types/model-performance';
import { PredictionRequest } from '../types/prediction-request';
import { PredictionResult } from '../types/prediction-result';
import { URLFeatures } from '../types/url-features';

@Injectable({
  providedIn: 'root'
})
export class PhishingApiService {
  private readonly apiUrl = `${environment.serverUrl}/api`;

  constructor(private http: HttpClient) {}

  public predictLogisticRegression(request: PredictionRequest): Observable<ApiResponse> {
    return this.http.post<ApiResponse>(`${this.apiUrl}/predict/logistic-regression`, request, environment.httpOptions);
  }

  public predictRandomForest(request: PredictionRequest): Observable<ApiResponse> {
    return this.http.post<ApiResponse>(`${this.apiUrl}/predict/random-forest`, request, environment.httpOptions);
  }

  public predictXGBoost(request: PredictionRequest): Observable<ApiResponse> {
    return this.http.post<ApiResponse>(`${this.apiUrl}/predict/xgboost`, request, environment.httpOptions);
  }

  public predictSVM(request: PredictionRequest): Observable<ApiResponse> {
    return this.http.post<ApiResponse>(`${this.apiUrl}/predict/svm`, request, environment.httpOptions);
  }

  public predictMLP(request: PredictionRequest): Observable<ApiResponse> {
    return this.http.post<ApiResponse>(`${this.apiUrl}/predict/mlp`, request, environment.httpOptions);
  }

  public predictAllAlgorithms(request: PredictionRequest): Observable<EnsembleResponse> {
    return this.http.post<EnsembleResponse>(`${this.apiUrl}/predict/all`, request, environment.httpOptions);
  }

  public predictEnsemble(features: URLFeatures): Observable<PredictionResult> {
    return this.http.post<PredictionResult>(`${this.apiUrl}/predict/ensemble`, features, environment.httpOptions);
  }

  public predictEnhanced(request: PredictionRequest & { model_type?: string }): Observable<ApiResponse> {
    return this.http.post<ApiResponse>(`${this.apiUrl}/predict/enhanced`, request, environment.httpOptions);
  }

  public getModelStatistics(modelName: string): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/statistics/model/${modelName}`, environment.httpOptions);
  }

  public getFeatureStatistics(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/statistics/features`, environment.httpOptions);
  }

  public getScalingStatistics(url: string): Observable<any> {
    const encodedUrl = encodeURIComponent(url);
    return this.http.get<any>(`${this.apiUrl}/statistics/scaling/${encodedUrl}`, environment.httpOptions);
  }

  public getAvailablePlots(): Observable<string[]> {
    return this.http.get<string[]>(`${this.apiUrl}/plots/list`, environment.httpOptions);
  }

  public getPlotUrl(filename: string): string {
    return `${this.apiUrl}/plots/${filename}`;
  }

  public getModelsPerformance(): Observable<ModelPerformance[]> {
    return this.http.get<ModelPerformance[]>(`${this.apiUrl}/models/performance`);
  }

  public getPredictionHistory(): Observable<any[]> {
    return this.http.get<any[]>(`${this.apiUrl}/results/history`);
  }
}
