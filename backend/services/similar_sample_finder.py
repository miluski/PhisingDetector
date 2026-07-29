import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

from utils.split_dataset_loader import feature_columns
from config.feature_modes import CANONICAL_MODES, normalize_mode

logger = logging.getLogger("SimilarSampleFinder")


class SimilarSampleFinder:
    """Find balanced nearest examples in configured training datasets."""

    def __init__(self):
        """Initialize lazy dataset caches and their canonical paths."""
        self.datasets: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {
            'current': {mode: None for mode in CANONICAL_MODES},
            'url_corpus': {mode: None for mode in CANONICAL_MODES}
        }
        self._dataset_paths = self._build_dataset_paths()

    def _build_dataset_paths(self):
        """Return paths for the current and repaired URL-corpus datasets."""
        base_path = Path(__file__).parent.parent / 'data'
        url_root = base_path / 'url_corpus'
        url_split_path = url_root / 'quality_v4_repaired_features'
        return {
            ('current', 'full'): base_path / 'dataset_full.csv',
            ('current', 'optimized'): base_path / 'dataset_optimized.csv',
            ('current', 'selected'): base_path / 'dataset_selected.csv',
            ('url_corpus', 'full'): url_split_path / 'dataset_full_train.csv',
            ('url_corpus', 'optimized'): url_split_path / 'dataset_optimized_train.csv',
            ('url_corpus', 'selected'): url_split_path / 'dataset_selected_train.csv',
        }

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
        mode_key = normalize_mode(mode)
        if mode_key not in CANONICAL_MODES:
            mode_key = 'full'
        if self.datasets[dataset_key][mode_key] is None:
            self._load_dataset_variant(
                self._dataset_paths[(dataset_key, mode_key)], dataset_key, mode_key
            )
        dataset_df = self.datasets[dataset_key][mode_key]
        logger.info(
            "Finding similar samples | mode=%s | dataset=%s | loaded=%s",
            mode_key,
            dataset_key,
            'yes' if dataset_df is not None else 'no'
        )
        return dataset_df

    def _get_available_features(self, dataset_df: pd.DataFrame, features: Dict[str, float]) -> List[str]:
        feature_cols = feature_columns(dataset_df)
        available_features = [f for f in feature_cols if f in features]
        logger.info("Available features: %d/%d", len(available_features), len(feature_cols))
        return available_features

    def _compute_distances(self, dataset_values: np.ndarray, sample_values: np.ndarray) -> np.ndarray:
        """Calculate missing-aware, min-max normalized Euclidean distances."""
        dataset_values, sample_values = self._as_float_arrays(dataset_values, sample_values)
        valid_mask = (dataset_values != -1) & (sample_values != -1)
        feature_min, feature_max = self._feature_bounds(dataset_values, valid_mask)
        feature_range = np.maximum(feature_max - feature_min, 1.0)
        dataset_normalized = (dataset_values - feature_min) / feature_range
        sample_normalized = (sample_values - feature_min) / feature_range
        differences = np.where(
            valid_mask,
            dataset_normalized - sample_normalized,
            0.0,
        )
        distances = np.sqrt(np.sum(differences ** 2, axis=1))
        distances[np.sum(valid_mask, axis=1) < 5] = np.inf
        return distances

    @staticmethod
    def _as_float_arrays(dataset_values, sample_values):
        """Convert dataset and query vectors to floating-point arrays."""
        return np.asarray(dataset_values, dtype=float), np.asarray(sample_values, dtype=float)

    @staticmethod
    def _feature_bounds(dataset_values, valid_mask):
        """Return finite per-feature bounds while ignoring missing values."""
        valid_values = np.where(valid_mask, dataset_values, np.nan)
        with np.errstate(all='ignore'):
            low = np.nanmin(valid_values, axis=0)
            high = np.nanmax(valid_values, axis=0)
        return np.nan_to_num(low, nan=0.0), np.nan_to_num(high, nan=1.0)

    def _try_add_result(self, results, row, idx, distance, available_features, limits) -> bool:
        """Append a candidate when its class quota is not exhausted."""
        is_phishing = int(row['phishing'])
        if is_phishing and limits['phishing'] >= limits['max_per_class']:
            return False
        if not is_phishing and limits['legitimate'] >= limits['max_per_class']:
            return False

        if is_phishing:
            limits['phishing'] += 1
        else:
            limits['legitimate'] += 1

        results.append(self._sample_payload(row, idx, distance, available_features))
        return True

    @staticmethod
    def _sample_payload(row, idx, distance, available_features):
        """Build the bounded response payload for one nearest sample."""
        values = {feature: float(row[feature]) for feature in available_features[:20]}
        return {'index': int(idx), 'distance': float(distance),
                'is_phishing': bool(int(row['phishing'])), 'features': values}

    def _select_results(self, dataset_df, available_features, distances, top_k) -> List[Dict]:
        """Select nearest results while keeping class representation balanced."""
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

    def find_similar_samples(self, features, mode='optimized', dataset='current', top_k=5) -> List[Dict]:
        """Return up to ``top_k`` missing-aware nearest dataset samples."""
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

    def get_feature_comparison(self, features, similar_sample, key_features=None) -> Dict:
        """Compare selected feature values with one similar sample."""
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
        """Return class counts and feature names for one current dataset mode."""
        dataset = self._get_dataset_df(mode, 'current')

        if dataset is None:
            return {}

        feature_cols = feature_columns(dataset)

        return {
            'total_samples': len(dataset),
            'phishing_samples': int(dataset['phishing'].sum()),
            'legitimate_samples': int(len(dataset) - dataset['phishing'].sum()),
            'feature_count': len(feature_cols),
            'features': feature_cols
        }
