"""HTTP endpoint exposing the reusable message example pool."""

from flask_restx import Resource

from app.api import core_ns
from app.dependencies import message_example_service
from services.message_example_service import serialize_message_examples


@core_ns.route("/message-examples")
class MessageExamplesResource(Resource):
    """Return e-mail and SMS examples for the interactive test form."""

    @core_ns.doc("message_examples")
    @core_ns.response(200, "Success")
    def get(self):
        """Read the configured example pool and return it as JSON."""
        return serialize_message_examples(message_example_service)
