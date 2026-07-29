"""HTTP resource for email analysis."""

from flask_restx import Resource
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import HTTPException

from app.api import core_ns, models
from app.dependencies import dataset_validator, mode_validator, model_validator, phishing_service
from app.request_parser import AnalysisRequestParser
from config.logging_config import get_logger

logger = get_logger("API.AnalyzeEmail")
EMAIL_PARSER = core_ns.parser()
EMAIL_PARSER.add_argument("sender", location="form")
EMAIL_PARSER.add_argument("receiver", location="form")
EMAIL_PARSER.add_argument("content", location="form", required=True)
EMAIL_PARSER.add_argument("mode", location="form")
EMAIL_PARSER.add_argument("dataset", location="form")
EMAIL_PARSER.add_argument("models", location="form", action="append")
EMAIL_PARSER.add_argument("attachments", type=FileStorage, location="files", action="append")


@core_ns.route("/analyze-email")
class AnalyzeEmailResource(Resource):
    """Validate and dispatch email analysis requests."""

    @core_ns.doc("analyze_email", consumes=["multipart/form-data"])
    @core_ns.expect(EMAIL_PARSER, validate=True)
    @core_ns.marshal_with(models["analysis_response"])
    @core_ns.response(200, "Success")
    @core_ns.response(400, "Validation Error")
    @core_ns.response(500, "Internal Server Error")
    def post(self):
        """Handle one email analysis request."""
        try:
            args = EMAIL_PARSER.parse_args()
            request = AnalysisRequestParser().parse_email(args, args.get("attachments"))
            self._validate(request)
            return phishing_service.analyze_email(**self._service_args(request))
        except HTTPException:
            raise
        except ValueError as exc:
            return self._error(str(exc), 400)
        except Exception as exc:
            logger.exception("Email analysis failed")
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
        return {"sender": request.sender, "receiver": request.receiver,
                "content": request.content, "attachments": request.attachments,
                "models": request.models, "mode": request.mode, "dataset": request.dataset}

    @staticmethod
    def _error(message: str, status: int):
        """Return the common API error tuple."""
        return {"success": False, "error": message}, status
