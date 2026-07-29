"""Stable response builders used by phishing analysis services."""

from typing import Dict, Any, List
from config.logging_config import get_logger

logger = get_logger("ResponseFactory")


class ResponseFactory:
    """Build success, empty-result and error response payloads."""

    @staticmethod
    def no_urls(message: str, dataset: str = 'current') -> Dict[str, Any]:
        """Build a successful response when no URL was extracted."""
        logger.info(f"   ℹ️ No URLs to analyze: {message}")
        return {
            'success': True,
            'is_phishing': False,
            'message': message,
            'dataset': dataset,
            'urls_found': 0,
            'urls': [],
            'analysis': []
        }

    @staticmethod
    def error(urls: List[str], error: str, dataset: str = 'current') -> Dict[str, Any]:
        """Build a failed response while retaining analyzed URL context."""
        logger.error(f"   ❌ Analysis error: {error}")
        return {
            'success': False,
            'error': error,
            'dataset': dataset,
            'urls_found': len(urls),
            'urls': urls
        }

    @staticmethod
    def success(
        urls: List[str], analysis: List[Dict], container_id: str,
        is_phishing: bool, dataset: str = 'current',
    ) -> Dict[str, Any]:
        """Build a successful response containing per-URL analysis results."""
        logger.info(f"   ✅ Analysis complete: {'PHISHING DETECTED' if is_phishing else 'SAFE'}")
        return {
            'success': True,
            'is_phishing': is_phishing,
            'dataset': dataset,
            'urls_found': len(urls),
            'urls': urls,
            'analysis': analysis,
            'container_id': container_id
        }
