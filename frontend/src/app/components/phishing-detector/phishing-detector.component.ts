import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { AVAILABLE_MODELS, PhishingAnalysisResponse } from '../../models/phishing-analysis.model';
import { PhishingDetectionService } from '../../services/phishing-detection.service';

@Component({
  selector: 'app-phishing-detector',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './phishing-detector.component.html',
  styleUrls: ['./phishing-detector.component.css'],
})
export class PhishingDetectorComponent {
  public analysisType: 'email' | 'sms' = 'email';
  public sender = '';
  public receiver = '';
  public content = '';
  public attachments: File[] = [];
  public mode: 'optimized' | 'full' = 'optimized';
  public availableModels = AVAILABLE_MODELS;
  public selectedModels: string[] = ['random_forest'];
  public isAnalyzing = false;
  public result: PhishingAnalysisResponse | null = null;
  public error: string | null = null;

  constructor(private phishingService: PhishingDetectionService) {}

  public toggleModel(modelId: string): void {
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
    const input = event.target as HTMLInputElement;
    if (input.files) {
      this.attachments = Array.from(input.files);
    }
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
    this.phishingService.analyzeSms(this.content, this.selectedModels, this.mode).subscribe({
      next: (response) => this.handleAnalysisSuccess(response),
      error: (err) => this.handleAnalysisError(err.error?.error || 'Analysis failed'),
    });
  }

  private analyzeEmailContent(): void {
    const formData = this.buildEmailForm();

    this.phishingService.analyzeEmail(formData).subscribe({
      next: (response) => this.handleAnalysisSuccess(response),
      error: (err) => this.handleAnalysisError(err.error?.error || 'Analiza nie powiodła się'),
    });
  }

  private buildEmailForm(): FormData {
    const formData = new FormData();
    formData.append('sender', this.sender);
    formData.append('receiver', this.receiver);
    formData.append('content', this.content);
    formData.append('mode', this.mode);
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
  }

  public getThreatLevel(): string {
    if (!this.result) return '';
    return this.result.is_phishing ? 'WYKRYTO PHISHING' : 'BEZPIECZNY';
  }

  public getThreatClass(): string {
    if (!this.result) return '';
    return this.result.is_phishing ? 'threat-high' : 'threat-safe';
  }

  public getModelDisplayName(modelId: string): string {
    const model = this.availableModels.find((m) => m.id === modelId);
    return model ? model.name : modelId;
  }

  public getFeatureEntries(features: Record<string, number>): { name: string; value: number }[] {
    return Object.entries(features).map(([name, value]) => ({ name, value }));
  }

  public getTopSampleFeatures(features: Record<string, number>): { name: string; value: number }[] {
    return Object.entries(features)
      .slice(0, 5)
      .map(([name, value]) => ({ name, value }));
  }

  public formatFeatureValue(value: number): string {
    if (typeof value !== 'number' || isNaN(value)) {
      return String(value);
    }
    return Number.isInteger(value) ? value.toString() : value.toFixed(4);
  }
}
