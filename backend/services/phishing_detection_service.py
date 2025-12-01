import uuid
from typing import Dict, Any, List
from extractors.content.url_extractor import URLExtractor
from services.container_orchestrator import ContainerOrchestrator
from services.analysis_visualizer import AnalysisVisualizer
from services.response_factory import ResponseFactory
from config.logging_config import get_logger
from config.operation_logger import OperationLogger

logger = get_logger("PhishingDetectionService")


class PhishingDetectionService:
    def __init__(self):
        logger.info("Initializing PhishingDetectionService...")
        self.url_extractor = URLExtractor()
        self.orchestrator = ContainerOrchestrator()
        self.visualizer = AnalysisVisualizer()
        logger.info("✅ PhishingDetectionService initialized")

    def _run_container_analysis(self, urls: List[str], models: List[str], mode: str) -> Dict[str, Any]:
        logger.info("   🐳 Starting container-based analysis...")
        container_id = self.orchestrator.create_container(urls, models, mode)
        logger.info(f"   ⏳ Container created, waiting for results...")
        result = self.orchestrator.wait_for_result(container_id, timeout=300)
        return {'container_id': container_id, 'result': result}

    def _is_phishing(self, analysis: List[Dict[str, Any]]) -> bool:
        if not analysis:
            return False
        return sum(1 for item in analysis if item.get('is_phishing', False)) > 0

    def analyze_email(
        self,
        sender: str,
        receiver: str,
        content: str,
        attachments: List[dict] = None,
        models: List[str] = None,
        mode: str = "optimized"
    ) -> Dict[str, Any]:
        op_logger = self._create_email_operation_logger(sender, receiver, content, attachments, models, mode)

        if models is None:
            models = ['random_forest']
            op_logger.step("Using default model: random_forest")

        urls = self._extract_email_urls(op_logger, sender, receiver, content, attachments)
        if not urls:
            op_logger.end_operation(success=True, result_summary="No URLs found in email")
            return ResponseFactory.no_urls('No URLs found in email')

        return self._analyze_urls(op_logger, urls, models, mode)

    def _create_email_operation_logger(self, sender, receiver, content, attachments, models, mode) -> OperationLogger:
        op_logger = OperationLogger(logger)
        op_logger.start_operation(
            str(uuid.uuid4())[:8],
            "EMAIL ANALYSIS",
            sender=sender or "(empty)",
            receiver=receiver or "(empty)",
            content_length=len(content),
            attachments=len(attachments) if attachments else 0,
            models=models,
            mode=mode
        )
        return op_logger

    def _extract_email_urls(self, op_logger, sender, receiver, content, attachments) -> List[str]:
        op_logger.step("Extracting URLs from email content...")
        urls = self.url_extractor.extract_all_urls(sender, receiver, content, attachments)
        if urls:
            op_logger.success(f"Found {len(urls)} URL(s)", urls=urls)
        return urls

    def analyze_sms(self, content: str, models: List[str] = None, mode: str = "optimized") -> Dict[str, Any]:
        op_logger = self._create_sms_operation_logger(content, models, mode)

        if models is None:
            models = ['random_forest']
            op_logger.step("Using default model: random_forest")

        urls = self._extract_sms_urls(op_logger, content)
        if not urls:
            op_logger.end_operation(success=True, result_summary="No URLs found in SMS")
            return ResponseFactory.no_urls('No URLs found in SMS')

        return self._analyze_urls(op_logger, urls, models, mode)

    def _create_sms_operation_logger(self, content, models, mode) -> OperationLogger:
        op_logger = OperationLogger(logger)
        op_logger.start_operation(
            str(uuid.uuid4())[:8],
            "SMS ANALYSIS",
            content_length=len(content),
            models=models,
            mode=mode
        )
        return op_logger

    def _extract_sms_urls(self, op_logger, content) -> List[str]:
        op_logger.step("Extracting URLs from SMS content...")
        urls = list(self.url_extractor.extract_from_text(content))
        if urls:
            op_logger.success(f"Found {len(urls)} URL(s)", urls=urls)
        return urls

    def _analyze_urls(self, op_logger, urls: List[str], models: List[str], mode: str) -> Dict[str, Any]:
        # Limit URLs to prevent timeout (max 15 URLs per analysis)
        MAX_URLS = 15
        if len(urls) > MAX_URLS:
            op_logger.warning(f"Too many URLs ({len(urls)}), limiting to first {MAX_URLS}")
            urls = urls[:MAX_URLS]

        container_id = None
        try:
            result_data = self._run_container_analysis(urls, models, mode)
            container_id = result_data['container_id']
            return self._process_container_result(op_logger, urls, result_data, container_id)
        except Exception as e:
            op_logger.error(f"Analysis failed", exception=e)
            op_logger.end_operation(success=False, result_summary=str(e))
            return ResponseFactory.error(urls, str(e))
        finally:
            self._cleanup_container(op_logger, container_id)

    def _process_container_result(self, op_logger, urls, result_data, container_id) -> Dict[str, Any]:
        result = result_data['result']
        if not result.get('success', False):
            error_msg = result.get('error', 'Container analysis failed')
            op_logger.error(f"Container analysis failed: {error_msg}")
            op_logger.end_operation(success=False, result_summary=error_msg)
            return ResponseFactory.error(urls, error_msg)

        return self._build_success_response(op_logger, urls, result, container_id)

    def _build_success_response(self, op_logger, urls, result, container_id) -> Dict[str, Any]:
        op_logger.step("Processing analysis results...")
        analysis = self._process_results(result, urls)

        op_logger.step("Generating visualizations...")
        is_phishing = self._is_phishing(analysis)
        response = ResponseFactory.success(urls, analysis, container_id, is_phishing)

        op_logger.end_operation(
            success=True,
            result_summary=f"{'PHISHING' if is_phishing else 'SAFE'} - {len(urls)} URL(s) analyzed"
        )
        return response

    def _cleanup_container(self, op_logger, container_id):
        if container_id:
            op_logger.step("Cleaning up container...")
            self.orchestrator.cleanup_container(container_id)

    def _process_results(self, result: Dict[str, Any], urls: List[str]) -> List[Dict[str, Any]]:
        return [self._process_url_result(url_result) for url_result in result.get('results', [])]

    def _process_url_result(self, url_result: Dict[str, Any]) -> Dict[str, Any]:
        if not url_result.get('success', False):
            return {'url': url_result.get('url'), 'success': False, 'error': url_result.get('error', 'Analysis failed')}
        return self._create_success_result(url_result)

    def _create_success_result(self, url_result: Dict[str, Any]) -> Dict[str, Any]:
        predictions = url_result.get('predictions', [])
        ensemble = url_result.get('ensemble', {})
        features = url_result.get('features', {})
        similar_samples = url_result.get('similar_samples', [])
        visualizations = self.visualizer.create_comprehensive_report(
            url=url_result['url'],
            predictions=predictions,
            ensemble_result=ensemble
        )
        return {
            'url': url_result['url'],
            'success': True,
            'predictions': predictions,
            'ensemble': ensemble,
            'features': features,
            'similar_samples': similar_samples,
            'visualizations': visualizations,
            'is_phishing': ensemble.get('ensemble_prediction') == 1
        }

    def cleanup(self):
        self.orchestrator.cleanup_all()
