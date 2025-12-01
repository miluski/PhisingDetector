from typing import Dict, Any, List
from config.logging_config import get_logger

logger = get_logger("ResponseFactory")


class ResponseFactory:
    @staticmethod
    def no_urls(message: str) -> Dict[str, Any]:
        logger.info(f"   ℹ️ No URLs to analyze: {message}")
        return {'success': True, 'is_phishing': False, 'message': message, 'urls_found': 0, 'urls': [], 'analysis': []}

    @staticmethod
    def error(urls: List[str], error: str) -> Dict[str, Any]:
        logger.error(f"   ❌ Analysis error: {error}")
        return {'success': False, 'error': error, 'urls_found': len(urls), 'urls': urls}

    @staticmethod
    def success(urls: List[str], analysis: List[Dict], container_id: str, is_phishing: bool) -> Dict[str, Any]:
        logger.info(f"   ✅ Analysis complete: {'PHISHING DETECTED' if is_phishing else 'SAFE'}")
        return {
            'success': True,
            'is_phishing': is_phishing,
            'urls_found': len(urls),
            'urls': urls,
            'analysis': analysis,
            'container_id': container_id
        }
