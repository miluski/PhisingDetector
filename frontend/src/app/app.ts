import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { PhishingDetectorComponent } from './components/phising-detector/phishing-detector.component';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, PhishingDetectorComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  protected readonly title = signal('PhishingDetector');
}
