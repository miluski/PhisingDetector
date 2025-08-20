import { Component, input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiResponse } from '../../types/api-response';
import { environment } from '../../../environments/environment.development';

@Component({
  selector: 'app-individual-result',
  imports: [CommonModule],
  templateUrl: './individual-result.html',
  styleUrl: './individual-result.scss',
  standalone: true
})
export class IndividualResult {
  protected individualResult = input.required<ApiResponse>();

  protected getFullPlotUrl(plotPath: string): string {
    return `${environment.serverUrl}${plotPath}`;
  }
}
