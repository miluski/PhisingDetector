"""OpenAPI response model definitions."""

from flask_restx import fields


class SwaggerModels:
    """Register response schemas shared by API resources."""

    @staticmethod
    def register_models(api):
        """Register analysis and health response models on an API instance."""
        analysis_models = SwaggerModels._create_analysis_models(api)
        response_models = SwaggerModels._create_response_models(api, analysis_models['url_analysis'])

        return response_models

    @staticmethod
    def _create_analysis_models(api):
        """Create nested prediction and URL-analysis models."""
        prediction = api.model('ModelPrediction', _prediction_fields())
        analysis = api.model('UrlAnalysis', _url_analysis_fields(prediction))
        return {'model_prediction': prediction, 'url_analysis': analysis}

    @staticmethod
    def _create_response_models(api, url_analysis_model):
        """Create top-level analysis and health response models."""
        analysis = api.model('AnalysisResponse', _response_fields(url_analysis_model))
        health = api.model('HealthResponse', _health_fields())
        return {'analysis_response': analysis, 'health_response': health}


def _prediction_fields():
    """Return OpenAPI fields for one model prediction."""
    return {
        'model': fields.String(description='Model name'),
        'mode': fields.String(description='Feature mode used by model'),
        'dataset': fields.String(description='Dataset used by model'),
        'prediction': fields.Integer(description='Prediction'),
        'prediction_label': fields.String(description='Human-readable prediction'),
        'confidence': fields.Float(description='Prediction confidence'),
        'probabilities': fields.Raw(description='Class probabilities'),
        'features_used': fields.Integer(description='Number of features'),
    }


def _url_analysis_fields(prediction):
    """Return OpenAPI fields for one URL analysis."""
    return {
        'url': fields.String(description='Analyzed URL'),
        'success': fields.Boolean(description='Analysis success status'),
        'error': fields.String(description='Error message if failed'),
        'predictions': fields.List(fields.Nested(prediction)),
        'ensemble': fields.Raw(description='Ensemble prediction result'),
        'features': fields.Raw(description='Extracted URL features'),
        'visualizations': fields.Raw(description='Visualization charts as base64'),
        'similar_samples': fields.Raw(description='Similar samples'),
        'is_phishing': fields.Boolean(description='Overall phishing verdict'),
        'raw_is_phishing': fields.Boolean(description='Raw verdict'),
        'receiver_domain': fields.Boolean(description='Receiver domain result'),
        'source': fields.String(description='URL source'),
        'classification_note': fields.String(description='Context classification note'),
    }


def _response_fields(url_analysis_model):
    """Return OpenAPI fields for the top-level analysis response."""
    return {
        'success': fields.Boolean(description='Request success status'),
        'is_phishing': fields.Boolean(description='Overall verdict'),
        'dataset': fields.String(description='Dataset used by models'),
        'message': fields.String(description='Informational message'),
        'error': fields.String(description='Error message'),
        'urls_found': fields.Integer(description='Number of URLs'),
        'urls': fields.List(fields.String, description='Analyzed URLs'),
        'analysis': fields.List(fields.Nested(url_analysis_model)),
        'container_id': fields.String(description='Sandbox container identifier'),
    }


def _health_fields():
    """Return OpenAPI fields for the health response."""
    return {
        'status': fields.String(description='Service health status'),
        'https': fields.Boolean(description='HTTPS enabled'),
    }
