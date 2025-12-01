from flask import redirect
from flask_restx import Resource
from app.api import root_ns


@root_ns.route('api-docs', doc=False)
class LegacyDocsResource(Resource):
    @root_ns.doc(False)
    def get(self):
        return redirect('/api/docs', code=302)
