import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger("SimilarSampleFinder")


class SimilarSampleFinder:
    def __init__(self):
        self.datasets: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {
            'current': {'full': None, 'optimized': None},
            'url_corpus': {'full': None, 'optimized': None}
        }
        self._load_datasets()

    def _load_datasets(self):
        base_path = Path(__file__).parent.parent / 'data'
        logger.info(f"Loading datasets from: {base_path}")

        self._load_dataset_variant(base_path / 'dataset_full.csv', 'current', 'full')
        self._load_dataset_variant(base_path / 'dataset_optimized.csv', 'current', 'optimized')
        self._load_dataset_variant(base_path / 'url_corpus' / 'dataset_full.csv', 'url_corpus', 'full')
        self._load_dataset_variant(base_path / 'url_corpus' / 'dataset_optimized.csv', 'url_corpus', 'optimized')

    def _load_dataset_variant(self, path: Path, dataset: str, mode: str):
        if path.exists():
            self.datasets[dataset][mode] = pd.read_csv(path)
            logger.info(f"Loaded {dataset}/{mode}: {len(self.datasets[dataset][mode])} samples")
            return
        logger.warning(f"Dataset not found: {path}")

    def _normalize_dataset(self, dataset: str) -> str:
        normalized = (dataset or 'current').strip().lower()
        return normalized if normalized in self.datasets else 'current'

    def _get_dataset_df(self, mode: str, dataset: str) -> Optional[pd.DataFrame]:
        dataset_key = self._normalize_dataset(dataset)
        mode_key = 'optimized' if mode == 'optimized' else 'full'
        dataset_df = self.datasets[dataset_key][mode_key]
        logger.info(
            "Finding similar samples | mode=%s | dataset=%s | loaded=%s",
            mode_key,
            dataset_key,
            'yes' if dataset_df is not None else 'no'
        )
        return dataset_df

    def _get_available_features(self, dataset_df: pd.DataFrame, features: Dict[str, float]) -> List[str]:
        feature_cols = [col for col in dataset_df.columns if col != 'phishing']
        available_features = [f for f in feature_cols if f in features]
        logger.info("Available features: %d/%d", len(available_features), len(feature_cols))
        return available_features

    def _compute_distances(self, dataset_values: np.ndarray, sample_values: np.ndarray) -> np.ndarray:
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

        return np.array(distances)

    def _try_add_result(
        self,
        results: List[Dict],
        row: pd.Series,
        idx: int,
        distance: float,
        available_features: List[str],
        limits: Dict[str, int]
    ) -> bool:
        is_phishing = int(row['phishing'])

        if is_phishing and limits['phishing'] >= limits['max_per_class']:
            return False
        if not is_phishing and limits['legitimate'] >= limits['max_per_class']:
            return False

        if is_phishing:
            limits['phishing'] += 1
        else:
            limits['legitimate'] += 1

        sample_data = {
            'index': int(idx),
            'distance': float(distance),
            'is_phishing': bool(is_phishing),
            'features': {}
        }

        for feature in available_features[:20]:
            sample_data['features'][feature] = float(row[feature])

        results.append(sample_data)
        return True

    def _select_results(
        self,
        dataset_df: pd.DataFrame,
        available_features: List[str],
        distances: np.ndarray,
        top_k: int
    ) -> List[Dict]:
        top_indices = np.argsort(distances)[:top_k * 2]
        results: List[Dict] = []
        limits = {
            'phishing': 0,
            'legitimate': 0,
            'max_per_class': (top_k + 1) // 2
        }

        for idx in top_indices:
            if len(results) >= top_k:
                break
            row = dataset_df.iloc[idx]
            self._try_add_result(results, row, idx, distances[idx], available_features, limits)

        return results

    def find_similar_samples(
        self,
        features: Dict[str, float],
        mode: str = 'optimized',
        dataset: str = 'current',
        top_k: int = 5
    ) -> List[Dict]:
        dataset_df = self._get_dataset_df(mode, dataset)

        if dataset_df is None:
            logger.warning("Dataset is None, returning empty list")
            return []

        available_features = self._get_available_features(dataset_df, features)

        if not available_features:
            logger.warning("No available features, returning empty list")
            return []

        sample_values = np.array([features.get(f, 0) for f in available_features])
        dataset_values = dataset_df[available_features].values
        distances = self._compute_distances(dataset_values, sample_values)
        results = self._select_results(dataset_df, available_features, distances, top_k)

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
        dataset = self.datasets['current']['optimized'] if mode == 'optimized' else self.datasets['current']['full']

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
