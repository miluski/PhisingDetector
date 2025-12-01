from flask_restx import Resource
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import HTTPException
from app.api import core_ns, models
from app.dependencies import mode_validator, phishing_service
from config.logging_config import get_logger

logger = get_logger("API.AnalyzeEmail")


EMAIL_PARSER = core_ns.parser()
EMAIL_PARSER.add_argument('sender', location='form')
EMAIL_PARSER.add_argument('receiver', location='form')
EMAIL_PARSER.add_argument('content', location='form', required=True)
EMAIL_PARSER.add_argument('mode', location='form')
EMAIL_PARSER.add_argument('models', location='form', action='append')
EMAIL_PARSER.add_argument('attachments', type=FileStorage, location='files', action='append')


@core_ns.route('/analyze-email')
class AnalyzeEmailResource(Resource):
    @core_ns.doc('analyze_email', consumes=['multipart/form-data'])
    @core_ns.expect(EMAIL_PARSER, validate=True)
    @core_ns.marshal_with(models['analysis_response'])
    @core_ns.response(200, 'Success')
    @core_ns.response(400, 'Validation Error')
    @core_ns.response(500, 'Internal Server Error')
    def post(self):
        logger.info("=" * 60)
        logger.info("📧 INCOMING EMAIL ANALYSIS REQUEST")
        logger.info("=" * 60)

        try:
            args = EMAIL_PARSER.parse_args()
            payload = self._parse_payload(args)

            logger.info(f"   Sender: {payload['sender'] or '(not provided)'}")
            logger.info(f"   Receiver: {payload['receiver'] or '(not provided)'}")
            logger.info(f"   Content length: {len(payload['content'])} chars")
            logger.info(f"   Mode: {payload['mode']}")
            logger.info(f"   Models: {payload['models']}")

            mode_validator.validate_or_raise(payload['mode'])
            attachments = self._prepare_attachments(args.get('attachments'))

            if attachments:
                logger.info(f"   Attachments: {len(attachments)} file(s)")
                for att in attachments:
                    logger.info(f"      - {att['filename']} ({len(att['content'])} bytes)")

            request_data = self._build_request(payload, attachments)
            logger.info("   ➜ Forwarding to PhishingDetectionService...")

            result = phishing_service.analyze_email(**request_data)

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
        return {
            'sender': self._clean_string(args.get('sender')),
            'receiver': self._clean_string(args.get('receiver')),
            'content': self._clean_string(args.get('content')),
            'mode': self._resolve_mode(args.get('mode')),
            'models': self._resolve_models(args.get('models'))
        }

    def _clean_string(self, value):
        return (value or '').strip()

    def _resolve_mode(self, mode):
        normalized = self._clean_string(mode)
        normalized = normalized.lower() if normalized else 'optimized'
        return normalized or 'optimized'

    def _resolve_models(self, raw):
        if not raw:
            return ['random_forest']
        items = raw if isinstance(raw, list) else [raw]
        filtered = [self._clean_string(item) for item in items if self._clean_string(item)]
        return filtered or ['random_forest']

    def _prepare_attachments(self, attachments):
        if not attachments:
            return None
        files = []
        for item in attachments:
            if item:
                files.append({'filename': item.filename, 'content': item.read()})
        return files or None

    def _build_request(self, payload, attachments):
        if not payload['content']:
            raise ValueError('Missing request body')
        return {
            'sender': payload['sender'],
            'receiver': payload['receiver'],
            'content': payload['content'],
            'attachments': attachments,
            'models': payload['models'],
            'mode': payload['mode']
        }

    def _error_response(self, message, status_code):
        return ({'success': False, 'error': message}, status_code)
