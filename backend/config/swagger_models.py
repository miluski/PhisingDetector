from flask_restx import fields


class SwaggerModels:
    @staticmethod
    def register_models(api):
        analysis_models = SwaggerModels._create_analysis_models(api)
        response_models = SwaggerModels._create_response_models(api, analysis_models['url_analysis'])

        return response_models

    @staticmethod
    def _create_analysis_models(api):
        model_prediction = api.model('ModelPrediction', {
            'model': fields.String(description='Model name'),
            'prediction': fields.Integer(description='Prediction (0=safe, 1=phishing)'),
            'prediction_label': fields.String(description='Human-readable prediction'),
            'confidence': fields.Float(description='Prediction confidence'),
            'probabilities': fields.Raw(description='Class probabilities')
        })

        url_analysis = api.model('UrlAnalysis', {
            'url': fields.String(description='Analyzed URL'),
            'success': fields.Boolean(description='Analysis success status'),
            'error': fields.String(description='Error message if failed'),
            'predictions': fields.List(fields.Nested(model_prediction)),
            'features': fields.Raw(description='Extracted URL features'),
            'is_phishing': fields.Boolean(description='Overall phishing verdict')
        })

        return {
            'model_prediction': model_prediction,
            'url_analysis': url_analysis
        }

    @staticmethod
    def _create_response_models(api, url_analysis_model):
        analysis_response = api.model('AnalysisResponse', {
            'success': fields.Boolean(description='Request success status'),
            'is_phishing': fields.Boolean(description='Overall phishing verdict'),
            'urls_found': fields.Integer(description='Number of URLs found'),
            'urls': fields.List(fields.String, description='List of analyzed URLs'),
            'analysis': fields.List(fields.Nested(url_analysis_model), description='Detailed analysis per URL')
        })

        health_response = api.model('HealthResponse', {
            'status': fields.String(description='Service health status'),
            'https': fields.Boolean(description='HTTPS enabled')
        })

        return {
            'analysis_response': analysis_response,
            'health_response': health_response
        }

