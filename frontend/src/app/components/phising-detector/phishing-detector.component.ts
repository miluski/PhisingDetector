import { CommonModule } from '@angular/common';
import { Component, WritableSignal, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { environment } from '../../../environments/environment.development';
import { PhishingApiService } from '../../services/phishing-api.service';
import { StatisticsPanel } from '../statistics-panel/statistics-panel';
import { ApiResponse } from '../../types/api-response';
import { FeatureStatistics, ModelStatistics, ScalingStatistics } from '../../types/enhanced-prediction';
import { EnsembleResponse } from '../../types/ensemble-response';
import { PredictionRequest } from '../../types/prediction-request';

@Component({
  selector: 'app-phishing-detector',
  imports: [CommonModule, FormsModule, StatisticsPanel],
  templateUrl: './phishing-detector.component.html',
  styleUrl: './phishing-detector.component.scss',
  standalone: true
})
export class PhishingDetectorComponent {
 
  protected messageType: WritableSignal<'email' | 'sms'> = signal('email');
  protected predictionRequest: PredictionRequest = {
    message_type: 'email',
    content: '',
    sender: '',
    receiver: ''
  };
  protected isLoading: WritableSignal<boolean> = signal(false);
  protected error: WritableSignal<string | null> = signal<string | null>(null);
  protected ensembleResult: WritableSignal<EnsembleResponse | null> = signal<EnsembleResponse | null>(null);
  protected individualResult: WritableSignal<ApiResponse | null> = signal<ApiResponse | null>(null);
  protected enhancedResult: WritableSignal<ApiResponse | null> = signal(null);
  protected modelStatistics: WritableSignal<ModelStatistics | null> = signal(null);
  protected featureStatistics: WritableSignal<FeatureStatistics | null> = signal(null);
  protected scalingStatistics: WritableSignal<ScalingStatistics | null> = signal(null);
  protected selectedModel: WritableSignal<string> = signal('logistic_regression');

  constructor(private phishingApi: PhishingApiService) {}

  protected onMessageTypeChange(type: 'email' | 'sms') {
    this.messageType.set(type);
    this.predictionRequest.message_type = type;
    
    this.predictionRequest.content = '';
    if (type === 'sms') {
      this.predictionRequest.sender = '';
      this.predictionRequest.receiver = '';
    }
    
    this.error.set(null);
    this.ensembleResult.set(null);
    this.individualResult.set(null);
  }

  protected onSubmit() {
    this.isLoading.set(true);
    this.error.set(null);
    this.ensembleResult.set(null);
    this.individualResult.set(null);
    this.phishingApi.predictAllAlgorithms(this.predictionRequest).subscribe({
      next: (response) => {
        this.ensembleResult.set(response);
        this.isLoading.set(false);
      },
      error: (err) => {
        this.error.set('Błąd podczas komunikacji z API: ' + (err.error?.error || err.message));
        this.isLoading.set(false);
      }
    });
  }

  protected testIndividualModel(model: string): void {
    this.isLoading.set(true);
    this.error.set(null);
    this.ensembleResult.set(null);
    this.individualResult.set(null);
    
    let apiCall;
    switch(model) {
      case 'logistic-regression':
        apiCall = this.phishingApi.predictLogisticRegression(this.predictionRequest);
        break;
      case 'random-forest':
        apiCall = this.phishingApi.predictRandomForest(this.predictionRequest);
        break;
      case 'xgboost':
        apiCall = this.phishingApi.predictXGBoost(this.predictionRequest);
        break;
      case 'svm':
        apiCall = this.phishingApi.predictSVM(this.predictionRequest);
        break;
      case 'mlp':
        apiCall = this.phishingApi.predictMLP(this.predictionRequest);
        break;
      default:
        this.error.set('Nieznany typ modelu');
        this.isLoading.set(false);
        return;
    }
    
    apiCall.subscribe({
      next: (response) => {
        this.individualResult.set(response);
        this.isLoading.set(false);
      },
      error: (err) => {
        this.error.set('Błąd podczas komunikacji z API: ' + (err.error?.error || err.message));
        this.isLoading.set(false);
      }
    });
  }

  protected getIndividualResults(): Array<any> {
    const result = this.ensembleResult();
    if (!result) return [];
    return [
      { name: 'Regresja Logistyczna', result: result.individual_predictions['logistic_regression'] },
      { name: 'Random Forest', result: result.individual_predictions['random_forest'] },
      { name: 'XGBoost', result: result.individual_predictions['xgboost'] },
      { name: 'SVM', result: result.individual_predictions['svm'] },
      { name: 'MLP', result: result.individual_predictions['mlp'] }
    ].filter(item => item.result);
  }

  protected getModelPlots(): Array<any> {
    const result = this.ensembleResult();
    if (!result?.plots) return [];
    return [
      { key: 'lr', name: 'Regresja Logistyczna', url: result.plots.logistic_regression },
      { key: 'rf', name: 'Random Forest', url: result.plots.random_forest },
      { key: 'xgb', name: 'XGBoost', url: result.plots.xgboost },
      { key: 'svm', name: 'SVM', url: result.plots.svm },
      { key: 'mlp', name: 'MLP', url: result.plots.mlp }
    ];
  }

  protected getRiskClass(riskLevel: string): string {
    return riskLevel.toLowerCase();
  }

  protected testEnhancedPrediction(): void {
    this.isLoading.set(true);
    this.error.set(null);
    this.enhancedResult.set(null);
    
    const request = { ...this.predictionRequest, model_type: this.selectedModel() };
    
    this.phishingApi.predictEnhanced(request).subscribe({
      next: (response) => {
        this.enhancedResult.set(response);
        this.isLoading.set(false);
      },
      error: (err) => {
        this.error.set('Błąd predykcji hybrydowej: ' + (err.error?.error || err.message));
        this.isLoading.set(false);
      }
    });
  }

  protected loadModelStatistics(): void {
    this.phishingApi.getModelStatistics(this.selectedModel()).subscribe({
      next: (stats) => {
        this.modelStatistics.set(stats);
      },
      error: (err) => {
        this.error.set('Błąd statystyk modelu: ' + (err.error?.error || err.message));
      }
    });
  }

  protected loadFeatureStatistics(): void {
    this.phishingApi.getFeatureStatistics().subscribe({
      next: (stats) => {
        this.featureStatistics.set(stats);
      },
      error: (err) => {
        this.error.set('Feature statistics error: ' + (err.error?.error || err.message));
      }
    });
  }

  protected loadScalingStatistics(): void {
    if (!this.predictionRequest.content || this.predictionRequest.content.trim() === '') {
      this.error.set('Please enter content with a URL to analyze scaling statistics');
      return;
    }

    const testUrl = this.extractUrlFromContent(this.predictionRequest.content);
    if (!testUrl) {
      this.error.set('No URL found in content for scaling analysis');
      return;
    }

    this.phishingApi.getScalingStatistics(testUrl).subscribe({
      next: (stats) => {
        this.scalingStatistics.set(stats);
      },
      error: (err) => {
        this.error.set('Scaling statistics error: ' + (err.error?.error || err.message));
      }
    });
  }

  private extractUrlFromContent(content: string): string | null {
    const urlRegex = /http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+/;
    const match = content.match(urlRegex);
    return match ? match[0] : null;
  }

  protected getModelOptions(): Array<{value: string, label: string}> {
    return [
      { value: 'logistic_regression', label: 'Logistic Regression' },
      { value: 'random_forest', label: 'Random Forest' },
      { value: 'xgboost', label: 'XGBoost' },
      { value: 'svm', label: 'SVM' },
      { value: 'mlp', label: 'MLP' }
    ];
  }

  protected onModelSelectionChanged(model: string): void {
    this.selectedModel.set(model);
  }

  protected getFullPlotUrl(relativePath: string): string {
    return `${environment.serverUrl}${relativePath}`;
  }
}
