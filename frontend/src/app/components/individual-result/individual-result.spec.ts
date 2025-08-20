import { ComponentFixture, TestBed } from '@angular/core/testing';

import { IndividualResult } from './individual-result';

describe('IndividualResult', () => {
  let component: IndividualResult;
  let fixture: ComponentFixture<IndividualResult>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [IndividualResult]
    })
    .compileComponents();

    fixture = TestBed.createComponent(IndividualResult);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
