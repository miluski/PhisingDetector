"""HTTP endpoint exposing lexical and network feature extraction."""

from flask_restx import Resource
from werkzeug.exceptions import HTTPException
from app.api import core_ns
from app.dependencies import feature_service, mode_validator, url_validator


EXTRACT_PARSER = core_ns.parser()
EXTRACT_PARSER.add_argument('url', required=True, location='form')
EXTRACT_PARSER.add_argument('mode', location='form')

@core_ns.route('/extract-features')
class ExtractFeaturesResource(Resource):
    """Validate a URL and return its extracted feature vector."""
    @core_ns.doc('extract_features', consumes=['multipart/form-data'])
    @core_ns.expect(EXTRACT_PARSER, validate=True)
    @core_ns.response(200, 'Success')
    @core_ns.response(400, 'Validation Error')
    @core_ns.response(500, 'Internal Server Error')
    def post(self):
        """Extract features and map validation or service errors to HTTP."""
        try:
            payload = self._parse_payload()
            self._validate_payload(payload)
            features = feature_service.extract(payload['url'], payload['mode'])
            return self._build_success(payload, features)
        except HTTPException as exc:
            raise exc
        except ValueError as exc:
            return self._error_response(str(exc), 400)
        except Exception as exc:
            return self._error_response(str(exc), 500)

    def _parse_payload(self):
        """Read and normalize URL and feature-mode form values."""
        args = EXTRACT_PARSER.parse_args()
        url_value = (args.get('url') or '').strip()
        mode_value = (args.get('mode') or 'optimized').strip().lower()
        return {'url': url_value, 'mode': mode_value or 'optimized'}

    def _validate_payload(self, payload):
        """Validate required URL content and supported feature mode."""
        if not payload['url']:
            raise ValueError('Missing required field: url')
        mode_validator.validate_or_raise(payload['mode'])
        url_validator.validate_or_raise(payload['url'])

    def _build_success(self, payload, features):
        """Build the stable successful endpoint response."""
        return {
            'success': True,
            'url': payload['url'],
            'mode': payload['mode'],
            'features': features,
            'feature_count': len(features)
        }

    def _error_response(self, message, status_code):
        """Build the stable error response tuple."""
        return ({'success': False, 'error': message}, status_code)
