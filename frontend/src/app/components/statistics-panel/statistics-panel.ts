import { CommonModule } from '@angular/common';
import { Component, OnInit, input, output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { FeatureStatistics, ModelStatistics, ScalingStatistics } from '../../types/enhanced-prediction';

interface ModelOption {
  value: string;
  label: string;
}

@Component({
  selector: 'app-statistics-panel',
  imports: [CommonModule, FormsModule],
  templateUrl: './statistics-panel.html',
  styleUrl: './statistics-panel.scss',
  standalone: true
})
export class StatisticsPanel implements OnInit {
  public modelStatistics = input<ModelStatistics | null>(null);
  public featureStatistics = input<FeatureStatistics | null>(null);
  public scalingStatistics = input<ScalingStatistics | null>(null);
  public selectedModel = input.required<string>();

  public onLoadModelStats = output<void>();
  public onLoadFeatureStats = output<void>();
  public onLoadScalingStats = output<void>();
  public onModelSelectionChanged = output<string>();

  protected modelOptions: ModelOption[] = [];

  public ngOnInit() {
    this.loadModelOptions();
  }

  private loadModelOptions() {
    this.modelOptions = [
      { value: 'logistic_regression', label: 'Regresja Logistyczna' },
      { value: 'random_forest', label: 'Random forest' },
      { value: 'xgboost', label: 'XGBoost' },
      { value: 'svm', label: 'SVM' },
      { value: 'mlp', label: 'MLP' }
    ];
  }

  protected onModelChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    this.onModelSelectionChanged.emit(target.value);
  }

  protected loadModelStatistics() {
    this.onLoadModelStats.emit();
  }

  protected loadFeatureStatistics() {
    this.onLoadFeatureStats.emit();
  }

  protected loadScalingStatistics() {
    this.onLoadScalingStats.emit();
  }

  protected getScalingKeys(): string[] {
    const stats = this.scalingStatistics();
    return stats ? Object.keys(stats) : [];
  }

  protected getScalingData(modelKey: string): any {
    const stats = this.scalingStatistics();
    return stats ? stats[modelKey] : null;
  }

  protected hasError(modelKey: string): boolean {
    const data = this.getScalingData(modelKey);
    return data && 'error' in data;
  }

  protected getError(modelKey: string): string {
    const data = this.getScalingData(modelKey);
    return data && 'error' in data ? data.error : '';
  }

  protected getExtremeValues(modelKey: string): string[] {
    const data = this.getScalingData(modelKey);
    return data && 'extreme_values' in data ? data.extreme_values : [];
  }

  protected getOriginalFeatures(modelKey: string): any {
    const data = this.getScalingData(modelKey);
    return data && 'original_features' in data ? data.original_features : {};
  }

  protected getScaledFeatures(modelKey: string): any {
    const data = this.getScalingData(modelKey);
    return data && 'scaled_features' in data ? data.scaled_features : {};
  }

  protected getScalingField(modelKey: string, field: string): any {
    const data = this.getScalingData(modelKey);
    return data && field in data ? data[field] : null;
  }

  protected getScalingFieldAsArray(modelKey: string, field: string): any[] {
    const data = this.getScalingData(modelKey);
    const fieldValue = data && field in data ? data[field] : null;
    return Array.isArray(fieldValue) ? fieldValue : [];
  }
}
