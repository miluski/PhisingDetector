import { Component, input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { EnsembleResponse } from '../../types/ensemble-response';
import { environment } from '../../../environments/environment.development';

@Component({
  selector: 'app-ensemble-results',
  imports: [CommonModule],
  templateUrl: './ensemble-results.html',
  styleUrl: './ensemble-results.scss',
  standalone: true
})
export class EnsembleResults {
  protected ensembleResult = input.required<EnsembleResponse>();

  protected getRiskClass(riskLevel: string): string {
    return riskLevel.toLowerCase();
  }

  protected getFullPlotUrl(plotPath: string): string {
    return `${environment.serverUrl}${plotPath}`;
  }

  protected getIndividualResults(): Array<{name: string, result: any}> {
    const result = this.ensembleResult();
    if (!result?.individual_predictions) return [];
    
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
}
