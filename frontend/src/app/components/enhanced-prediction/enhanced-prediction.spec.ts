import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EnhancedPrediction } from './enhanced-prediction';

describe('EnhancedPrediction', () => {
  let component: EnhancedPrediction;
  let fixture: ComponentFixture<EnhancedPrediction>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EnhancedPrediction]
    })
    .compileComponents();

    fixture = TestBed.createComponent(EnhancedPrediction);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
