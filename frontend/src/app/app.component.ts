import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { PhishingDetectorComponent } from './components/phishing-detector/phishing-detector.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, PhishingDetectorComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent {
  title = 'PhishingDetector';
}
