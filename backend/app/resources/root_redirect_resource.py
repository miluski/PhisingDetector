"""Root redirect for the API service."""

from flask import redirect
from flask_restx import Resource
from app.api import root_ns


@root_ns.route('', doc=False)
class RootRedirectResource(Resource):
    """Redirect the root URL to the legacy documentation route."""
    @root_ns.doc(False)
    def get(self):
        """Return the root redirect response."""
        return redirect('/api-docs', code=302)
