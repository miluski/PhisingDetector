"""Health endpoint for liveness checks."""

from flask_restx import Resource
from app.api import core_ns, models


@core_ns.route('/health')
class HealthResource(Resource):
    """Report that the HTTP application is alive."""
    @core_ns.doc('health_check')
    @core_ns.marshal_with(models['health_response'])
    def get(self):
        """Return the liveness payload."""
        return {'status': 'healthy', 'https': True}
