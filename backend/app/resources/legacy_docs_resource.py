"""Compatibility redirect for the historical API documentation URL."""

from flask import redirect
from flask_restx import Resource
from app.api import root_ns


@root_ns.route('api-docs', doc=False)
class LegacyDocsResource(Resource):
    """Redirect the old documentation route to the current route."""
    @root_ns.doc(False)
    def get(self):
        """Return a permanent application-level documentation redirect."""
        return redirect('/api/docs', code=302)
