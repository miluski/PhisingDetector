from flask_restx import Resource
from werkzeug.exceptions import HTTPException
from app.api import core_ns, models
from app.dependencies import mode_validator, dataset_validator, phishing_service
from config.logging_config import get_logger

logger = get_logger("API.AnalyzeSMS")


SMS_PARSER = core_ns.parser()
SMS_PARSER.add_argument('content', location='form', required=True)
SMS_PARSER.add_argument('models', location='form', action='append')
SMS_PARSER.add_argument('mode', location='form')
SMS_PARSER.add_argument('dataset', location='form')

@core_ns.route('/analyze-sms')
class AnalyzeSmsResource(Resource):
    @core_ns.doc('analyze_sms', consumes=['multipart/form-data'])
    @core_ns.expect(SMS_PARSER, validate=True)
    @core_ns.marshal_with(models['analysis_response'])
    @core_ns.response(200, 'Success')
    @core_ns.response(400, 'Validation Error')
    @core_ns.response(500, 'Internal Server Error')
    def post(self):
        logger.info("=" * 60)
        logger.info("💬 INCOMING SMS ANALYSIS REQUEST")
        logger.info("=" * 60)

        try:
            args = SMS_PARSER.parse_args()
            payload = self._parse_payload(args)

            logger.info(f"   Content length: {len(payload['content'])} chars")
            logger.info(f"   Mode: {payload['mode']}")
            logger.info(f"   Dataset: {payload['dataset']}")
            logger.info(f"   Models: {payload['models']}")

            mode_validator.validate_or_raise(payload['mode'])
            dataset_validator.validate_or_raise(payload['dataset'])

            logger.info("   ➜ Forwarding to PhishingDetectionService...")
            result = phishing_service.analyze_sms(**payload)

            logger.info(f"   ✅ Request completed: {'PHISHING' if result.get('is_phishing') else 'SAFE'}")
            return result
        except HTTPException as exc:
            logger.warning(f"   ⚠️ HTTP Exception: {exc}")
            raise exc
        except ValueError as exc:
            logger.warning(f"   ⚠️ Validation Error: {exc}")
            return self._error_response(str(exc), 400)
        except Exception as exc:
            logger.error(f"   ❌ Internal Error: {exc}")
            return self._error_response(str(exc), 500)

    def _parse_payload(self, args):
        content = self._clean_string(args.get('content'))
        if not content:
            raise ValueError('Missing required field: content')
        return {
            'content': content,
            'models': self._resolve_models(args.get('models')),
            'mode': self._resolve_mode(args.get('mode')),
            'dataset': self._resolve_dataset(args.get('dataset'))
        }

    def _clean_string(self, value):
        return (value or '').strip()

    def _resolve_models(self, raw):
        if not raw:
            return ['random_forest']
        items = raw if isinstance(raw, list) else [raw]
        filtered = [self._clean_string(item) for item in items if self._clean_string(item)]
        return filtered or ['random_forest']

    def _resolve_mode(self, mode):
        normalized = self._clean_string(mode)
        normalized = normalized.lower() if normalized else 'optimized'
        return normalized or 'optimized'

    def _resolve_dataset(self, dataset):
        normalized = self._clean_string(dataset)
        normalized = normalized.lower() if normalized else 'current'
        return normalized or 'current'

    def _error_response(self, message, status_code):
        return ({'success': False, 'error': message}, status_code)
