import { DATASET_FEATURE_COUNTS } from './phishing-analysis.model';

describe('DATASET_FEATURE_COUNTS', () => {
  it('matches active URL Corpus artifacts', () => {
    expect(DATASET_FEATURE_COUNTS.url_corpus).toEqual({
      optimized: 47,
      full: 110,
    });
  });

  it('keeps published baseline artifact sizes', () => {
    expect(DATASET_FEATURE_COUNTS.current).toEqual({
      optimized: 74,
      full: 111,
    });
  });
});
