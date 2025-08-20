import { Component, input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiResponse } from '../../types/api-response';
import { EnhancedPrediction } from '../../types/enhanced-prediction';

@Component({
  selector: 'app-enhanced-prediction',
  imports: [CommonModule],
  templateUrl: './enhanced-prediction.html',
  styleUrl: './enhanced-prediction.scss'
})
export class EnhancedPredictionComponent {
  protected enhancedResult = input.required<ApiResponse>();

  protected getRiskClass(riskLevel: string): string {
    return riskLevel.toLowerCase();
  }

  protected getEnhancedPrediction(): EnhancedPrediction {
    return this.enhancedResult().prediction as EnhancedPrediction;
  }
}
