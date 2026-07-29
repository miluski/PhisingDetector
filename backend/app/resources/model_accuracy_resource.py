"""HTTP endpoint exposing accuracies saved by the model notebooks."""

from flask_restx import Resource

from app.api import core_ns
from services.model_accuracy_service import serialize_accuracy_response
from app.dependencies import model_accuracy_service


@core_ns.route("/model-accuracies")
class ModelAccuracyResource(Resource):
    """Return current test accuracy values for all available models."""

    @core_ns.doc("model_accuracies")
    @core_ns.response(200, "Success")
    def get(self):
        """Read saved comparison CSV files and return their accuracy cells."""
        return serialize_accuracy_response(model_accuracy_service)
