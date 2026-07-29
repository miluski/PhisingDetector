"""HTTP resource for SMS analysis."""

from flask_restx import Resource
from werkzeug.exceptions import HTTPException

from app.api import core_ns, models
from app.dependencies import dataset_validator, mode_validator, model_validator, phishing_service
from app.request_parser import AnalysisRequestParser
from config.logging_config import get_logger

logger = get_logger("API.AnalyzeSMS")
SMS_PARSER = core_ns.parser()
SMS_PARSER.add_argument("content", location="form", required=True)
SMS_PARSER.add_argument("models", location="form", action="append")
SMS_PARSER.add_argument("mode", location="form")
SMS_PARSER.add_argument("dataset", location="form")


@core_ns.route("/analyze-sms")
class AnalyzeSmsResource(Resource):
    """Validate and dispatch SMS analysis requests."""

    @core_ns.doc("analyze_sms", consumes=["multipart/form-data"])
    @core_ns.expect(SMS_PARSER, validate=True)
    @core_ns.marshal_with(models["analysis_response"])
    @core_ns.response(200, "Success")
    @core_ns.response(400, "Validation Error")
    @core_ns.response(500, "Internal Server Error")
    def post(self):
        """Handle one SMS analysis request."""
        try:
            request = AnalysisRequestParser().parse_sms(SMS_PARSER.parse_args())
            self._validate(request)
            return phishing_service.analyze_sms(**self._service_args(request))
        except HTTPException:
            raise
        except ValueError as exc:
            return self._error(str(exc), 400)
        except Exception as exc:
            logger.exception("SMS analysis failed")
            return self._error(str(exc), 500)

    @staticmethod
    def _validate(request):
        """Validate normalized request values."""
        mode_validator.validate_or_raise(request.mode)
        dataset_validator.validate_or_raise(request.dataset)
        model_validator.validate_or_raise(request.models)

    @staticmethod
    def _service_args(request):
        """Convert the request dataclass to service keyword arguments."""
        return {"content": request.content, "models": request.models,
                "mode": request.mode, "dataset": request.dataset}

    @staticmethod
    def _error(message: str, status: int):
        """Return the common API error tuple."""
        return {"success": False, "error": message}, status
