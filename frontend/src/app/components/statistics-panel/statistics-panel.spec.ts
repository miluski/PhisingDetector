import { ComponentFixture, TestBed } from '@angular/core/testing';

import { StatisticsPanel } from './statistics-panel';

describe('StatisticsPanel', () => {
  let component: StatisticsPanel;
  let fixture: ComponentFixture<StatisticsPanel>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [StatisticsPanel]
    })
    .compileComponents();

    fixture = TestBed.createComponent(StatisticsPanel);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
