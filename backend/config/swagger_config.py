from flask_restx import Api


class SwaggerConfig:
    @staticmethod
    def create_api(app):
        return Api(
            app,
            version='1.0',
            title='PhishingDetector API',
            description='Advanced ML-based phishing detection system for emails and SMS',
            doc='/api/docs',
            add_specs=False
        )
