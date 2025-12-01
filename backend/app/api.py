from flask_restx import Namespace
from app import app
from config.swagger_config import SwaggerConfig
from config.swagger_models import SwaggerModels

api = SwaggerConfig.create_api(app)
models = SwaggerModels.register_models(api)

root_ns = Namespace('root', description='Redirect endpoints')
core_ns = Namespace('api', description='Phishing detection operations')

api.add_namespace(root_ns, path='/')
api.add_namespace(core_ns, path='/api')
