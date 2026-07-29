import os
import json
import sys
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stderr
)
logger = logging.getLogger("SandboxRunner")


class SandboxRunner:
    """Execute URL feature extraction and model prediction inside a sandbox."""
    def __init__(self):
        self.start_time = datetime.now()
        self.feature_service = None
        self.prediction_service = None

    def _load_config(self) -> Tuple[List[str], List[str], str, str]:
        urls = json.loads(os.environ.get('URLS', '[]'))
        models = json.loads(os.environ.get('MODELS', '[]'))
        mode = os.environ.get('MODE', 'optimized')
        dataset = os.environ.get('DATASET', 'current')
        return urls, models, mode, dataset

    def _log_config(self, urls: List[str], models: List[str], mode: str, dataset: str):
        logger.info("📋 Configuration:")
        logger.info(f"   Mode: {mode}")
        logger.info(f"   Dataset: {dataset}")
        logger.info(f"   Models: {models}")
        logger.info(f"   URLs to analyze: {len(urls)}")
        for i, url in enumerate(urls, 1):
            logger.info(f"      {i}. {url}")

    def _initialize_services(self):
        logger.info("-" * 60)
        logger.info("🔧 Initializing services...")
        from services.url_feature_service import URLFeatureService
        from services.ml_prediction_service import MLPredictionService
        self.feature_service = URLFeatureService(apply_network_defaults=False)
        self.prediction_service = MLPredictionService()
        logger.info("   ✅ Services initialized")

    def _extract_features(self, url: str, mode: str) -> Dict[str, Any]:
        logger.info("   📊 Extracting features...")
        features = self.feature_service.extract(url, mode)
        logger.info(f"   ✅ Extracted {len(features)} features")
        return features

    def _run_predictions(self, features: Dict, models: List[str], mode: str, dataset: str) -> tuple:
        logger.info(f"   🤖 Running predictions with {len(models)} model(s)...")
        for model_name in models:
            logger.info(f"      - Loading model: {model_name}")
        predictions, similar_samples = self.prediction_service.predict_multiple_models(features, models, mode, dataset)
        self._log_predictions(predictions)
        return predictions, similar_samples

    def _log_predictions(self, predictions: List[Dict]):
        logger.info("   ✅ Predictions complete")
        for pred in predictions:
            if 'error' in pred:
                logger.warning(f"      {pred['model']}: ❌ ERROR - {pred['error']}")
                continue
            status = "🚨 PHISHING" if pred['prediction'] == 1 else "✅ SAFE"
            logger.info(f"      {pred['model']}: {status} (confidence: {pred['confidence']*100:.1f}%)")

    def _compute_ensemble(
        self,
        predictions: List[Dict],
        dataset: str,
        similar_samples: List[Dict] | None = None,
    ) -> Dict:
        logger.info("   🎯 Computing ensemble prediction...")
        ensemble = self.prediction_service.get_ensemble_from_predictions(
            predictions, dataset, similar_samples
        )
        self._log_ensemble(ensemble)
        return ensemble

    def _log_ensemble(self, ensemble: Dict):
        status = "🚨 PHISHING" if ensemble['ensemble_prediction'] == 1 else "✅ SAFE"
        logger.info(f"   ✅ Ensemble result: {status}")
        voting = ensemble['models_voting']
        logger.info(
            f"      Voting: {voting['phishing']} phishing / {voting['safe']} safe"
        )
        logger.info(f"      Confidence: {ensemble['confidence']*100:.1f}%")

    def _analyze_url(self, url: str, idx: int, total: int, models: List[str], mode: str, dataset: str) -> Dict:
        logger.info("-" * 60)
        logger.info(f"🔍 ANALYZING URL {idx}/{total}: {url}")
        url_result = {'url': url, 'success': False}

        try:
            url_result.update(self._analyze_payload(url, models, mode, dataset))
        except Exception as e:
            logger.exception("   ❌ Error analyzing URL")
            url_result['error'] = str(e)

        return url_result

    def _analyze_payload(self, url, models, mode, dataset):
        """Extract, predict and assemble one successful URL result."""
        features = self._extract_features(url, mode)
        predictions, similar = self._run_predictions(features, models, mode, dataset)
        ensemble = self._compute_ensemble(predictions, dataset, similar)
        return {'success': True, 'features': features, 'predictions': predictions,
                'ensemble': ensemble, 'similar_samples': similar}

    def _log_completion(self, results: List[Dict], urls: List[str]):
        duration = (datetime.now() - self.start_time).total_seconds()
        successful = sum(1 for r in results if r.get('success'))
        logger.info("=" * 60)
        logger.info("✅ SANDBOX ANALYSIS COMPLETE")
        logger.info(f"   Duration: {duration:.2f}s")
        logger.info(f"   URLs processed: {len(urls)}")
        logger.info(f"   Successful: {successful}/{len(urls)}")
        logger.info("=" * 60)

    def run(self) -> Dict[str, Any]:
        """Run configured URLs and return a JSON-serializable result."""
        logger.info("=" * 60)
        logger.info("🔒 SANDBOX CONTAINER STARTED")
        logger.info("=" * 60)

        urls, models, mode, dataset = self._load_config()
        self._log_config(urls, models, mode, dataset)
        self._initialize_services()

        results = self._run_urls(urls, models, mode, dataset)
        self._log_completion(results, urls)

        return self._result_payload(results, urls, mode, dataset, models)

    def _result_payload(self, results, urls, mode, dataset, models):
        """Build the public sandbox response dictionary."""
        return {'success': True, 'results': results, 'total_urls': len(urls),
                'mode': mode, 'dataset': dataset, 'models': models}

    def _run_urls(self, urls, models, mode, dataset):
        """Analyze configured URLs in input order."""
        return [self._analyze_url(url, index, len(urls), models, mode, dataset)
                for index, url in enumerate(urls, 1)]


def main():
    """Run the sandbox runner and emit a controlled JSON error on failure."""
    try:
        runner = SandboxRunner()
        output = runner.run()
        print(json.dumps(output))
        sys.exit(0)
    except Exception as e:
        logger.exception("❌ SANDBOX CRITICAL ERROR")
        print(json.dumps({'success': False, 'error': str(e)}))
        sys.exit(1)


if __name__ == '__main__':
    main()
