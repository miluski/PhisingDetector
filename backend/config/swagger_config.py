"""OpenAPI configuration for the Flask-RESTX application."""

from flask_restx import Api


class SwaggerConfig:
    """Construct the public API documentation object."""

    @staticmethod
    def create_api(app):
        """Create and configure Flask-RESTX API documentation."""
        return Api(
            app,
            version='1.0',
            title='PhishingDetector API',
            description='Advanced ML-based phishing detection system for emails and SMS',
            doc='/api/docs',
            add_specs=False
        )
