import { CommonModule } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
  AVAILABLE_DATASETS,
  AVAILABLE_MODELS,
  DATASET_FEATURE_COUNTS,
  EMPTY_MODEL_TEST_ACCURACY,
  EmailExample,
  AnalysisMode,
  AnalysisType,
  DatasetType,
  ModelId,
  MessageExamplesResponse,
  PhishingAnalysisResponse,
} from '../../models/phishing-analysis.model';
import { PhishingDetectionService } from '../../services/phishing-detection.service';

@Component({
  selector: 'app-phishing-detector',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './phishing-detector.component.html',
  styleUrls: ['./phishing-detector.component.css'],
})
export class PhishingDetectorComponent {
  public analysisType: AnalysisType = 'email';
  public sender = '';
  public receiver = '';
  public content = '';
  public attachments: File[] = [];
  public mode: AnalysisMode = 'optimized';
  public dataset: DatasetType = 'current';
  public readonly availableModels = AVAILABLE_MODELS;
  public readonly availableDatasets = AVAILABLE_DATASETS;
  public readonly datasetFeatureCounts = DATASET_FEATURE_COUNTS;
  public modelTestAccuracy = EMPTY_MODEL_TEST_ACCURACY;
  public isLoadingAccuracies = true;
  public messageExamples: MessageExamplesResponse = {
    success: false,
    emails: [],
    sms: [],
  };
  public isLoadingExamples = true;
  public selectedExampleIsPhishing: boolean | null = null;
  public selectedModels: ModelId[] = ['random_forest'];
  public isAnalyzing = false;
  public result: PhishingAnalysisResponse | null = null;
  public error: string | null = null;

  constructor(private readonly phishingService: PhishingDetectionService) {
    this.loadModelAccuracies();
    this.loadMessageExamples();
  }

  private loadModelAccuracies(): void {
    this.phishingService.getModelAccuracies().subscribe({
      next: (response) => {
        this.modelTestAccuracy = response.accuracies;
        this.isLoadingAccuracies = false;
      },
      error: () => {
        this.isLoadingAccuracies = false;
      },
    });
  }

  private loadMessageExamples(): void {
    this.phishingService.getMessageExamples().subscribe({
      next: (response) => {
        this.messageExamples = response;
        this.isLoadingExamples = false;
      },
      error: () => {
        this.isLoadingExamples = false;
      },
    });
  }

  public toggleModel(modelId: ModelId): void {
    const index = this.selectedModels.indexOf(modelId);
    if (index > -1) {
      this.selectedModels.splice(index, 1);
    } else {
      this.selectedModels.push(modelId);
    }
  }

  public selectAllModels(): void {
    this.selectedModels = this.availableModels.map((m) => m.id);
  }

  public deselectAllModels(): void {
    this.selectedModels = [];
  }

  public onFileSelected(event: Event): void {
    const input = event.target;
    if (!(input instanceof HTMLInputElement) || input.files === null) return;
    this.attachments = Array.from(input.files);
  }

  public analyze(): void {
    if (!this.validateInput()) {
      return;
    }

    this.prepareAnalysis();

    if (this.analysisType === 'sms') {
      this.analyzeSmsContent();
    } else {
      this.analyzeEmailContent();
    }
  }

  private validateInput(): boolean {
    if (this.selectedModels.length === 0) {
      this.error = 'Proszę wybrać co najmniej jeden model ML';
      return false;
    }

    if (!this.content.trim()) {
      this.error = 'Proszę podać treść do analizy';
      return false;
    }

    return true;
  }

  private prepareAnalysis(): void {
    this.isAnalyzing = true;
    this.result = null;
    this.error = null;
  }

  private analyzeSmsContent(): void {
    this.phishingService
      .analyzeSms(this.content, this.selectedModels, this.mode, this.dataset)
      .subscribe({
        next: (response) => this.handleAnalysisSuccess(response),
        error: (error: unknown) => this.handleAnalysisError(
          this.extractErrorMessage(error, 'Analiza nie powiodła się'),
        ),
      });
  }

  private analyzeEmailContent(): void {
    const formData = this.buildEmailForm();

    this.phishingService.analyzeEmail(formData).subscribe({
      next: (response) => this.handleAnalysisSuccess(response),
      error: (error: unknown) => this.handleAnalysisError(
        this.extractErrorMessage(error, 'Analiza nie powiodła się'),
      ),
    });
  }

  private buildEmailForm(): FormData {
    const formData = new FormData();
    formData.append('sender', this.sender);
    formData.append('receiver', this.receiver);
    formData.append('content', this.content);
    formData.append('mode', this.mode);
    formData.append('dataset', this.dataset);
    this.selectedModels.forEach((model) => formData.append('models', model));
    this.attachments.forEach((file) => formData.append('attachments', file, file.name));
    return formData;
  }

  private handleAnalysisSuccess(response: PhishingAnalysisResponse): void {
    if (!response.success && response.error) {
      this.handleAnalysisError(response.error);
      return;
    }
    this.result = response;
    this.isAnalyzing = false;
  }

  private handleAnalysisError(errorMessage: string): void {
    this.error = errorMessage;
    this.isAnalyzing = false;
  }

  public reset(): void {
    this.sender = '';
    this.receiver = '';
    this.content = '';
    this.attachments = [];
    this.result = null;
    this.error = null;
    this.selectedExampleIsPhishing = null;
  }

  public getThreatLevel(): string {
    if (!this.result) return '';
    return this.result.is_phishing ? 'WYKRYTO PHISHING' : 'BEZPIECZNY';
  }

  public getThreatClass(): string {
    if (!this.result) return '';
    return this.result.is_phishing ? 'threat-high' : 'threat-safe';
  }

  public getModelDisplayName(modelId: ModelId | string): string {
    const model = this.availableModels.find((m) => m.id === modelId);
    return model ? model.name : modelId;
  }

  public getFeatureCount(mode: AnalysisMode = this.mode): number {
    return this.datasetFeatureCounts[this.dataset][mode];
  }

  public getModelAccuracy(modelId: ModelId): string {
    const accuracy = this.modelTestAccuracy[this.dataset][this.mode][modelId];
    if (this.isLoadingAccuracies) return 'Wczytywanie...';
    return accuracy === undefined ? 'Brak aktualnego wyniku' : `${(accuracy * 100).toFixed(2)}%`;
  }

  public insertExample(): void {
    const examples = this.analysisType === 'email'
      ? this.messageExamples.emails
      : this.messageExamples.sms;
    if (examples.length === 0) {
      return;
    }
    const example = examples[Math.floor(Math.random() * examples.length)];
    this.selectedExampleIsPhishing = example.contains_phishing_url;
    if (this.analysisType === 'email') {
      this.insertEmailExample(example as EmailExample);
      return;
    }
    this.content = example.body;
  }

  public hasMessageExamples(): boolean {
    const examples = this.analysisType === 'email'
      ? this.messageExamples.emails
      : this.messageExamples.sms;
    return examples.length > 0;
  }

  public clearExampleStatus(): void {
    this.selectedExampleIsPhishing = null;
  }

  private insertEmailExample(example: EmailExample): void {
    this.sender = example.sender;
    this.receiver = example.recipient;
    this.content = [
      example.subject ? `Temat: ${example.subject}` : '',
      example.body,
    ].filter(Boolean).join('\n\n');
  }

  public getFeatureEntries(features: Record<string, number>): { name: string; value: number }[] {
    return Object.entries(features).map(([name, value]) => ({ name, value }));
  }

  public getTopSampleFeatures(features: Record<string, number>): { name: string; value: number }[] {
    return this.getFeatureEntries(features).slice(0, 5);
  }

  public formatFeatureValue(value: number): string {
    if (typeof value !== 'number' || Number.isNaN(value)) {
      return String(value);
    }
    return Number.isInteger(value) ? value.toString() : value.toFixed(4);
  }

  private extractErrorMessage(error: unknown, fallback: string): string {
    if (error instanceof HttpErrorResponse && this.isErrorPayload(error.error)) {
      return error.error.error;
    }
    return fallback;
  }

  private isErrorPayload(value: unknown): value is { error: string } {
    return typeof value === 'object' && value !== null
      && 'error' in value && typeof value.error === 'string';
  }
}
