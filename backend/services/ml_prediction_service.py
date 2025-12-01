import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
import logging
from services.similar_sample_finder import SimilarSampleFinder

logger = logging.getLogger("MLPredictionService")


class MLPredictionService:
    MODELS_WITH_MODEL_SUFFIX = {'xgboost', 'mlp', 'svm'}
    MODELS_REQUIRING_SCALER = {'mlp', 'svm', 'logistic_regression'}

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.models_dir = Path(__file__).parent.parent / "models"
        self.similar_sample_finder = SimilarSampleFinder()
        logger.info(f"MLPredictionService initialized | Models dir: {self.models_dir}")

    def _get_model_filename(self, model_name: str, mode: str) -> str:
        if model_name in self.MODELS_WITH_MODEL_SUFFIX:
            return f"{model_name}_model_{mode}.pkl"
        return f"{model_name}_{mode}.pkl"

    def _get_scaler_filename(self, model_name: str, mode: str) -> str:
        return f"{model_name}_scaler_{mode}.pkl"

    def _load_scaler(self, model_name: str, mode: str):
        scaler_key = f"{model_name}_scaler_{mode}"
        if scaler_key in self.scalers:
            return self.scalers[scaler_key]

        scaler_path = self.models_dir / model_name / self._get_scaler_filename(model_name, mode)
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            self.scalers[scaler_key] = scaler
            logger.info(f"   ✅ Scaler loaded: {scaler_key}")
            return scaler
        return None

    def load_model(self, model_name: str, mode: str = "optimized"):
        model_key = f"{model_name}_{mode}"

        if model_key in self.models:
            logger.debug(f"   📦 Model cached: {model_key}")
            return self.models[model_key]

        model_file = self._get_model_filename(model_name, mode)
        model_path = self.models_dir / model_name / model_file
        logger.info(f"   📥 Loading model: {model_path}")

        if not model_path.exists():
            logger.error(f"   ❌ Model not found: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, 'rb') as f:
            loaded_data = pickle.load(f)

        model_data = self._normalize_model_data(loaded_data, model_name, mode)
        self.models[model_key] = model_data
        logger.info(f"   ✅ Model loaded: {model_key} ({len(model_data.get('feature_names', []))} features)")
        return model_data

    def _normalize_model_data(self, loaded_data, model_name: str, mode: str) -> Dict[str, Any]:
        if isinstance(loaded_data, dict) and 'model' in loaded_data:
            return loaded_data

        scaler = self._load_scaler(model_name, mode)
        feature_names = self._get_feature_names_from_scaler_or_model(scaler, loaded_data, model_name, mode)

        return {
            'model': loaded_data,
            'scaler': scaler,
            'feature_names': feature_names
        }

    def _get_feature_names_from_scaler_or_model(self, scaler, model, model_name: str, mode: str) -> List[str]:
        if scaler is not None and hasattr(scaler, 'feature_names_in_'):
            return list(scaler.feature_names_in_)
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        from config.feature_names_config import FeatureConfig
        return FeatureConfig.get_feature_names(mode)

    def _prepare_feature_vector(self, features: Dict[str, float], feature_names: List[str]) -> pd.DataFrame:
        feature_values = {name: features.get(name, 0.0) for name in feature_names}
        df = pd.DataFrame([feature_values], columns=feature_names)
        df = df.fillna(0.0)
        return df

    def _build_prediction_result(
        self,
        model_name: str,
        mode: str,
        prediction: int,
        prediction_proba: np.ndarray,
        feature_count: int
    ) -> Dict[str, Any]:
        return {
            'model': model_name,
            'mode': mode,
            'prediction': int(prediction),
            'prediction_label': 'Phishing' if prediction == 1 else 'Safe',
            'probabilities': {
                'safe': float(prediction_proba[0]),
                'phishing': float(prediction_proba[1])
            },
            'confidence': float(max(prediction_proba)),
            'features_used': feature_count
        }

    def predict_url(
        self,
        features: Dict[str, float],
        model_name: str,
        mode: str = "optimized",
        include_similar: bool = True
    ) -> Dict[str, Any]:
        model_data = self.load_model(model_name, mode)
        model = model_data['model']
        feature_names = model_data['feature_names']
        scaler = model_data.get('scaler')

        feature_vector = self._prepare_feature_vector(features, feature_names)

        if scaler is not None and model_name in self.MODELS_REQUIRING_SCALER:
            feature_vector = pd.DataFrame(
                scaler.transform(feature_vector),
                columns=feature_names
            )

        prediction, prediction_proba = self._get_prediction_and_proba(model, feature_vector, model_name)

        result = self._build_prediction_result(
            model_name, mode, prediction, prediction_proba, len(feature_names)
        )

        if model_name == 'random_forest':
            result.update(
                self._analyze_random_forest(model, feature_vector, features)
            )

        if include_similar:
            similar_samples = self.similar_sample_finder.find_similar_samples(
                features, mode, top_k=3
            )
            result['similar_samples'] = similar_samples

        return result

    def _get_prediction_and_proba(self, model, feature_vector: pd.DataFrame, model_name: str):
        if model_name == 'mlp':
            X = feature_vector.values
            proba = model.predict(X, verbose=0)[0]
            if len(proba) == 1:
                phishing_prob = float(proba[0])
                prediction_proba = np.array([1 - phishing_prob, phishing_prob])
            else:
                prediction_proba = proba
            prediction = 1 if prediction_proba[1] > 0.5 else 0
        else:
            X = feature_vector.values
            prediction = model.predict(X)[0]
            prediction_proba = model.predict_proba(X)[0]
        return prediction, prediction_proba

    def _calculate_tree_votes(self, model, feature_vector: pd.DataFrame) -> Dict[str, int]:
        X = feature_vector.values
        tree_predictions = np.array([
            tree.predict(X)[0]
            for tree in model.estimators_
        ])

        return {
            'safe': int(np.sum(tree_predictions == 0)),
            'phishing': int(np.sum(tree_predictions == 1)),
            'total': len(model.estimators_)
        }

    def _extract_top_features(self, model, feature_vector: pd.DataFrame, features: Dict[str, float]) -> List[Dict[str, Any]]:
        feature_importance = model.feature_importances_
        feature_names = feature_vector.columns.tolist()
        sorted_indices = np.argsort(feature_importance)[::-1]

        return [
            {
                'name': feature_names[idx],
                'importance': float(feature_importance[idx]),
                'value': float(features.get(feature_names[idx], 0))
            }
            for idx in sorted_indices[:10]
        ]

    def _analyze_random_forest(
        self,
        model,
        feature_vector: pd.DataFrame,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        votes = self._calculate_tree_votes(model, feature_vector)
        top_features = self._extract_top_features(model, feature_vector, features)

        return {
            'voting': {
                'safe': votes['safe'],
                'phishing': votes['phishing'],
                'total_trees': votes['total'],
                'safe_percentage': float(votes['safe'] / votes['total'] * 100),
                'phishing_percentage': float(votes['phishing'] / votes['total'] * 100)
            },
            'top_features': top_features
        }

    def predict_multiple_models(
        self,
        features: Dict[str, float],
        model_names: List[str],
        mode: str = "optimized"
    ) -> List[Dict[str, Any]]:
        results = []
        similar_samples = None

        for i, model_name in enumerate(model_names):
            try:
                include_similar = (i == 0)
                result = self.predict_url(features, model_name, mode, include_similar)

                if include_similar and 'similar_samples' in result:
                    similar_samples = result.pop('similar_samples')

                results.append(result)
            except Exception as e:
                results.append({
                    'model': model_name,
                    'mode': mode,
                    'error': str(e),
                    'success': False
                })

        return results, similar_samples

    def _filter_successful_predictions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [p for p in predictions if 'error' not in p]

    def _calculate_ensemble_votes(self, predictions: List[Dict[str, Any]]) -> Dict[str, int]:
        phishing_count = sum(1 for p in predictions if p['prediction'] == 1)
        return {
            'phishing': phishing_count,
            'safe': len(predictions) - phishing_count,
            'total': len(predictions)
        }

    def _calculate_avg_probability(self, predictions: List[Dict[str, Any]]) -> float:
        return np.mean([p['probabilities']['phishing'] for p in predictions])

    def get_ensemble_prediction(
        self,
        features: Dict[str, float],
        model_names: List[str],
        mode: str = "optimized"
    ) -> Dict[str, Any]:
        predictions, similar_samples = self.predict_multiple_models(features, model_names, mode)
        successful = self._filter_successful_predictions(predictions)

        if not successful:
            return {'success': False, 'error': 'All models failed to predict'}

        votes = self._calculate_ensemble_votes(successful)
        avg_prob = self._calculate_avg_probability(successful)
        ensemble_pred = 1 if votes['phishing'] > votes['safe'] else 0

        result = {
            'ensemble_prediction': int(ensemble_pred),
            'ensemble_label': 'Phishing' if ensemble_pred == 1 else 'Safe',
            'models_voting': votes,
            'average_phishing_probability': float(avg_prob),
            'confidence': float(max(votes['phishing'], votes['safe']) / votes['total']),
            'individual_predictions': successful
        }

        if similar_samples:
            result['similar_samples'] = similar_samples

        return result
