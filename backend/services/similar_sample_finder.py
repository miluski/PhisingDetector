import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger("SimilarSampleFinder")


class SimilarSampleFinder:
    def __init__(self):
        self.dataset_full: Optional[pd.DataFrame] = None
        self.dataset_optimized: Optional[pd.DataFrame] = None
        self._load_datasets()

    def _load_datasets(self):
        base_path = Path(__file__).parent.parent / 'data'
        logger.info(f"Loading datasets from: {base_path}")

        full_path = base_path / 'dataset_full.csv'
        if full_path.exists():
            self.dataset_full = pd.read_csv(full_path)
            logger.info(f"Loaded dataset_full.csv: {len(self.dataset_full)} samples")
        else:
            logger.warning(f"Dataset not found: {full_path}")

        optimized_path = base_path / 'dataset_optimized.csv'
        if optimized_path.exists():
            self.dataset_optimized = pd.read_csv(optimized_path)
            logger.info(f"Loaded dataset_optimized.csv: {len(self.dataset_optimized)} samples")
        else:
            logger.warning(f"Dataset not found: {optimized_path}")

    def find_similar_samples(
        self,
        features: Dict[str, float],
        mode: str = 'optimized',
        top_k: int = 5
    ) -> List[Dict]:
        dataset = self.dataset_optimized if mode == 'optimized' else self.dataset_full
        logger.info(f"Finding similar samples | mode={mode} | dataset={'loaded' if dataset is not None else 'None'}")

        if dataset is None:
            logger.warning("Dataset is None, returning empty list")
            return []

        feature_cols = [col for col in dataset.columns if col != 'phishing']
        available_features = [f for f in feature_cols if f in features]
        logger.info(f"Available features: {len(available_features)}/{len(feature_cols)}")

        if not available_features:
            logger.warning("No available features, returning empty list")
            return []

        sample_values = np.array([features.get(f, 0) for f in available_features])
        dataset_values = dataset[available_features].values

        sample_valid = sample_values != -1
        dataset_valid = dataset_values != -1

        distances = []
        for i, row in enumerate(dataset_values):
            valid_mask = sample_valid & dataset_valid[i]
            if np.sum(valid_mask) < 5:
                distances.append(float('inf'))
                continue

            sample_subset = sample_values[valid_mask]
            row_subset = row[valid_mask]

            sample_norm = self._normalize(sample_subset)
            row_norm = self._normalize(row_subset)

            dist = np.sqrt(np.sum((sample_norm - row_norm) ** 2))
            distances.append(dist)

        distances = np.array(distances)
        top_indices = np.argsort(distances)[:top_k * 2]

        results = []
        phishing_count = 0
        legitimate_count = 0
        max_per_class = (top_k + 1) // 2

        for idx in top_indices:
            if len(results) >= top_k:
                break

            row = dataset.iloc[idx]
            is_phishing = int(row['phishing'])

            if is_phishing and phishing_count >= max_per_class:
                continue
            if not is_phishing and legitimate_count >= max_per_class:
                continue

            if is_phishing:
                phishing_count += 1
            else:
                legitimate_count += 1

            sample_data = {
                'index': int(idx),
                'distance': float(distances[idx]),
                'is_phishing': bool(is_phishing),
                'features': {}
            }

            for feature in available_features[:20]:
                sample_data['features'][feature] = float(row[feature])

            results.append(sample_data)

        logger.info(f"Found {len(results)} similar samples")
        return results

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val - min_val == 0:
            return np.zeros_like(values)
        return (values - min_val) / (max_val - min_val)

    def get_feature_comparison(
        self,
        features: Dict[str, float],
        similar_sample: Dict,
        key_features: List[str] = None
    ) -> Dict:
        if key_features is None:
            key_features = [
                'qty_dot_url', 'qty_hyphen_url', 'length_url', 'domain_length',
                'qty_redirects', 'time_response', 'tls_ssl_certificate',
                'qty_dot_domain', 'qty_hyphen_domain', 'directory_length'
            ]

        comparison = {}
        for feature in key_features:
            if feature in features and feature in similar_sample.get('features', {}):
                comparison[feature] = {
                    'current': features[feature],
                    'similar': similar_sample['features'][feature],
                    'diff': abs(features[feature] - similar_sample['features'][feature])
                }

        return comparison

    def get_dataset_statistics(self, mode: str = 'optimized') -> Dict:
        dataset = self.dataset_optimized if mode == 'optimized' else self.dataset_full

        if dataset is None:
            return {}

        feature_cols = [col for col in dataset.columns if col != 'phishing']

        return {
            'total_samples': len(dataset),
            'phishing_samples': int(dataset['phishing'].sum()),
            'legitimate_samples': int(len(dataset) - dataset['phishing'].sum()),
            'feature_count': len(feature_cols),
            'features': feature_cols
        }
