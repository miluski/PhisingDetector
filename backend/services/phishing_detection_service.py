import uuid
from typing import Dict, Any, List, Set
from extractors.content.url_extractor import URLExtractor
from services.container_orchestrator import ContainerOrchestrator
from services.analysis_visualizer import AnalysisVisualizer
from services.response_factory import ResponseFactory
from config.logging_config import get_logger
from config.operation_logger import OperationLogger

logger = get_logger("PhishingDetectionService")


class PhishingDetectionService:
    """Coordinate extraction, sandbox inference and response presentation."""

    def __init__(self, url_extractor=None, orchestrator=None, visualizer=None):
        """Initialize collaborators, allowing tests to inject fakes."""
        logger.info("Initializing PhishingDetectionService...")
        self.url_extractor = url_extractor or URLExtractor()
        self.orchestrator = orchestrator or ContainerOrchestrator()
        self.visualizer = visualizer or AnalysisVisualizer()
        logger.info("✅ PhishingDetectionService initialized")

    def _run_container_analysis(self, urls: List[str], models: List[str], mode: str, dataset: str) -> Dict[str, Any]:
        logger.info("   🐳 Starting container-based analysis...")
        container_id = self.orchestrator.create_container(urls, models, mode, dataset)
        logger.info("   ⏳ Container created, waiting for results...")
        result = self.orchestrator.wait_for_result(container_id, timeout=300)
        return {'container_id': container_id, 'result': result}

    def _is_phishing(self, analysis: List[Dict[str, Any]]) -> bool:
        if not analysis:
            return False
        return sum(1 for item in analysis if item.get('is_phishing', False)) > 0

    def analyze_email(self, sender, receiver, content, attachments=None, models=None,
                      mode="optimized", dataset='current') -> Dict[str, Any]:
        """Extract URLs from an email and analyze them in the sandbox."""
        op_logger = self._create_email_operation_logger(sender, receiver, content, attachments, models, mode, dataset)
        models = self._ensure_models(models, op_logger)

        receiver_urls = set(self.url_extractor.extract_from_email_address(receiver))
        message_urls = set(self.url_extractor.extract_all_urls(sender, None, content, attachments))
        receiver_urls.difference_update(message_urls)
        urls = self._extract_email_urls(op_logger, sender, receiver, content, attachments)
        if not urls:
            op_logger.end_operation(success=True, result_summary="No URLs found in email")
            return ResponseFactory.no_urls('No URLs found in email', dataset)

        return self._analyze_urls(op_logger, urls, models, mode, dataset, receiver_urls)

    def _create_email_operation_logger(
        self, sender, receiver, content, attachments, models, mode, dataset,
    ) -> OperationLogger:
        """Create an operation logger containing email context metadata."""
        op_logger = OperationLogger(logger)
        op_logger.start_operation(
            str(uuid.uuid4())[:8],
            "EMAIL ANALYSIS",
            sender=sender or "(empty)",
            receiver=receiver or "(empty)",
            content_length=len(content),
            attachments=len(attachments) if attachments else 0,
            models=models,
            mode=mode,
            dataset=dataset
        )
        return op_logger

    def _extract_email_urls(self, op_logger, sender, receiver, content, attachments) -> List[str]:
        op_logger.step("Extracting URLs from email content...")
        urls = self.url_extractor.extract_all_urls(sender, receiver, content, attachments)
        if urls:
            op_logger.success(f"Found {len(urls)} URL(s)", urls=urls)
        return urls

    def analyze_sms(self, content, models=None, mode="optimized", dataset='current') -> Dict[str, Any]:
        """Extract URLs from an SMS and analyze them in the sandbox."""
        op_logger = self._create_sms_operation_logger(content, models, mode, dataset)
        models = self._ensure_models(models, op_logger)

        urls = self._extract_sms_urls(op_logger, content)
        if not urls:
            op_logger.end_operation(success=True, result_summary="No URLs found in SMS")
            return ResponseFactory.no_urls('No URLs found in SMS', dataset)

        return self._analyze_urls(op_logger, urls, models, mode, dataset)

    @staticmethod
    def _ensure_models(models, op_logger):
        """Apply the public default model when no model selection was provided."""
        if models is not None:
            return models
        op_logger.step("Using default model: random_forest")
        return ['random_forest']

    def _create_sms_operation_logger(self, content, models, mode, dataset) -> OperationLogger:
        op_logger = OperationLogger(logger)
        op_logger.start_operation(
            str(uuid.uuid4())[:8],
            "SMS ANALYSIS",
            content_length=len(content),
            models=models,
            mode=mode,
            dataset=dataset
        )
        return op_logger

    def _extract_sms_urls(self, op_logger, content) -> List[str]:
        op_logger.step("Extracting URLs from SMS content...")
        urls = list(self.url_extractor.extract_from_text(content))
        if urls:
            op_logger.success(f"Found {len(urls)} URL(s)", urls=urls)
        return urls

    def _analyze_urls(self, op_logger, urls, models, mode, dataset, receiver_domain_urls=None) -> Dict[str, Any]:
        """Run one bounded URL batch and always clean up its container."""
        urls = self._limit_urls(op_logger, urls)
        container_id = None
        try:
            result_data = self._run_container_analysis(urls, models, mode, dataset)
            container_id = result_data['container_id']
            return self._process_container_result(
                op_logger, urls, result_data, container_id, dataset, receiver_domain_urls or set()
            )
        except Exception as e:
            op_logger.error("Analysis failed", exception=e)
            op_logger.end_operation(success=False, result_summary=str(e))
            return ResponseFactory.error(urls, str(e), dataset)
        finally:
            self._cleanup_container(op_logger, container_id)

    @staticmethod
    def _limit_urls(op_logger, urls):
        """Limit user-controlled URL batches to the sandbox safety bound."""
        limit = 15
        if len(urls) <= limit:
            return urls
        op_logger.warning(f"Too many URLs ({len(urls)}), limiting to first {limit}")
        return urls[:limit]

    def _process_container_result(
        self, op_logger, urls, result_data, container_id, dataset: str,
        receiver_domain_urls: Set[str]
    ) -> Dict[str, Any]:
        result = result_data['result']
        if not result.get('success', False):
            error_msg = result.get('error', 'Container analysis failed')
            op_logger.error(f"Container analysis failed: {error_msg}")
            op_logger.end_operation(success=False, result_summary=error_msg)
            return ResponseFactory.error(urls, error_msg, dataset)

        return self._build_success_response(
            op_logger, urls, result, container_id, dataset, receiver_domain_urls
        )

    def _build_success_response(
        self, op_logger, urls, result, container_id, dataset: str,
        receiver_domain_urls: Set[str]
    ) -> Dict[str, Any]:
        op_logger.step("Processing analysis results...")
        analysis = self._process_results(result, receiver_domain_urls)

        op_logger.step("Generating visualizations...")
        is_phishing = self._is_phishing(analysis)
        response = ResponseFactory.success(urls, analysis, container_id, is_phishing, dataset)

        op_logger.end_operation(
            success=True,
            result_summary=f"{'PHISHING' if is_phishing else 'SAFE'} - {len(urls)} URL(s) analyzed"
        )
        response['dataset'] = dataset
        return response

    def _cleanup_container(self, op_logger, container_id):
        if container_id:
            op_logger.step("Cleaning up container...")
            self.orchestrator.cleanup_container(container_id)

    def _process_results(self, result: Dict[str, Any], receiver_domain_urls: Set[str] = None) -> List[Dict[str, Any]]:
        receiver_domain_urls = receiver_domain_urls or set()
        return [
            self._process_url_result(url_result, url_result.get('url') in receiver_domain_urls)
            for url_result in result.get('results', [])
        ]

    def _process_url_result(self, url_result: Dict[str, Any], receiver_domain: bool = False) -> Dict[str, Any]:
        if not url_result.get('success', False):
            return {
                'url': url_result.get('url'),
                'success': False,
                'error': url_result.get('error', 'Analysis failed'),
                'receiver_domain': receiver_domain,
                'source': 'receiver_email' if receiver_domain else 'message_content',
            }
        return self._create_success_result(url_result, receiver_domain)

    def _create_success_result(self, url_result: Dict[str, Any], receiver_domain: bool = False) -> Dict[str, Any]:
        """Build one successful URL result with visualizations and context."""
        predictions = url_result.get('predictions', [])
        ensemble = url_result.get('ensemble', {})
        features = url_result.get('features', {})
        similar_samples = url_result.get('similar_samples', [])
        visualizations = self.visualizer.create_comprehensive_report(
            url=url_result['url'],
            predictions=predictions,
            ensemble_result=ensemble
        )
        context_note = (
            'Domena odbiorcy analizowana przez modele. Jej wynik wpływa na ensemble.'
            if receiver_domain else None
        )
        return self._result_payload(
            url_result, predictions, ensemble, features, similar_samples,
            visualizations, receiver_domain, context_note,
        )

    @staticmethod
    def _result_payload(
        url_result, predictions, ensemble, features, similar_samples,
        visualizations, receiver_domain, context_note,
    ):
        """Return the stable successful per-URL response structure."""
        return {
            'url': url_result['url'],
            'success': True,
            'predictions': predictions,
            'ensemble': ensemble,
            'features': features,
            'similar_samples': similar_samples,
            'visualizations': visualizations,
            'is_phishing': ensemble.get('ensemble_prediction') == 1,
            'raw_is_phishing': ensemble.get('ensemble_prediction') == 1,
            'receiver_domain': receiver_domain,
            'source': 'receiver_email' if receiver_domain else 'message_content',
            'classification_note': context_note,
        }

    def cleanup(self):
        """Clean up all containers owned by the orchestrator."""
        self.orchestrator.cleanup_all()
