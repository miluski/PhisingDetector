import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EnsembleResults } from './ensemble-results';

describe('EnsembleResults', () => {
  let component: EnsembleResults;
  let fixture: ComponentFixture<EnsembleResults>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EnsembleResults]
    })
    .compileComponents();

    fixture = TestBed.createComponent(EnsembleResults);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
