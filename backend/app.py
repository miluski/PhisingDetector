from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
import os
import json
from datetime import datetime
from services.logistic_regression_service import LogisticRegressionService
from services.random_forest_service import RandomForestService
from services.xgboost_service import XGBoostService
from services.svm_service import SVMService
from services.mlp_service import MLPService

app = Flask(__name__)
CORS(app)

PLOTS_DIR = "static/plots"
RESULTS_DIR = "static/results"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

lr_service = LogisticRegressionService()
rf_service = RandomForestService()
xgb_service = XGBoostService()
svm_service = SVMService()
mlp_service = MLPService()

@app.route('/api/predict/logistic-regression', methods=['POST'])
def predict_logistic_regression():
    try:
        data = request.get_json()
        if not data or 'message_type' not in data or 'content' not in data:
            return jsonify({"error": "Missing required fields: message_type and content"}), 400
        
        message_type = data.get('message_type', '')
        content = data.get('content', '')
        sender = data.get('sender', '') if message_type == 'email' else ''
        receiver = data.get('receiver', '') if message_type == 'email' else ''
        
        if message_type == 'email' and not sender:
            return jsonify({"error": "Sender email is required for email messages"}), 400
        
        result = lr_service.predict(sender, content, receiver, message_type)
        if 'error' in result:
            return jsonify(result), 500
        return jsonify({
            "algorithm": "Logistic Regression",
            "prediction": result,
            "input": {
                "message_type": message_type,
                "sender": sender if message_type == 'email' else "N/A",
                "receiver": receiver if receiver else "Not provided",
                "content_length": len(content)
            },
            "plots": {
                "analysis": "/api/plots/logistic_regression_analysis.png",
                "comparison": "/api/plots/comprehensive_model_comparison.png"
            }
        })
    except Exception as e:
        logger.error(f"Error in logistic regression prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/predict/random-forest', methods=['POST'])
def predict_random_forest():
    try:
        data = request.get_json()
        if not data or 'message_type' not in data or 'content' not in data:
            return jsonify({"error": "Missing required fields: message_type and content"}), 400
        
        message_type = data.get('message_type', '')
        content = data.get('content', '')
        sender = data.get('sender', '') if message_type == 'email' else ''
        receiver = data.get('receiver', '') if message_type == 'email' else ''
        
        if message_type == 'email' and not sender:
            return jsonify({"error": "Sender email is required for email messages"}), 400
        
        result = rf_service.predict(sender, content, receiver, message_type)
        if 'error' in result:
            return jsonify(result), 500
        return jsonify({
            "algorithm": "Random Forest",
            "prediction": result,
            "input": {
                "message_type": message_type,
                "sender": sender if message_type == 'email' else "N/A",
                "receiver": receiver if receiver else "Not provided",
                "content_length": len(content)
            },
            "plots": {
                "analysis": "/api/plots/random_forest_analysis.png",
                "comparison": "/api/plots/comprehensive_model_comparison.png"
            }
        })
    except Exception as e:
        logger.error(f"Error in random forest prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/predict/xgboost', methods=['POST'])
def predict_xgboost():
    try:
        data = request.get_json()
        if not data or 'message_type' not in data or 'content' not in data:
            return jsonify({"error": "Missing required fields: message_type and content"}), 400
        
        message_type = data.get('message_type', '')
        content = data.get('content', '')
        sender = data.get('sender', '') if message_type == 'email' else ''
        receiver = data.get('receiver', '') if message_type == 'email' else ''
        
        if message_type == 'email' and not sender:
            return jsonify({"error": "Sender email is required for email messages"}), 400
        
        result = xgb_service.predict(sender, content, receiver, message_type)
        if 'error' in result:
            return jsonify(result), 500 
        return jsonify({
            "algorithm": "XGBoost",
            "prediction": result,
            "input": {
                "message_type": message_type,
                "sender": sender if message_type == 'email' else "N/A",
                "receiver": receiver if receiver else "Not provided",
                "content_length": len(content)
            },
            "plots": {
                "analysis": "/api/plots/xgboost_analysis.png",
                "comparison": "/api/plots/comprehensive_model_comparison.png"
            }
        })
    except Exception as e:
        logger.error(f"Error in XGBoost prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/predict/svm', methods=['POST'])
def predict_svm():
    try:
        data = request.get_json()
        if not data or 'message_type' not in data or 'content' not in data:
            return jsonify({"error": "Missing required fields: message_type and content"}), 400
        
        message_type = data.get('message_type', '')
        content = data.get('content', '')
        sender = data.get('sender', '') if message_type == 'email' else ''
        receiver = data.get('receiver', '') if message_type == 'email' else ''
        
        if message_type == 'email' and not sender:
            return jsonify({"error": "Sender email is required for email messages"}), 400
        
        result = svm_service.predict(sender, content, receiver, message_type)
        if 'error' in result:
            return jsonify(result), 500
        return jsonify({
            "algorithm": "SVM",
            "prediction": result,
            "input": {
                "message_type": message_type,
                "sender": sender if message_type == 'email' else "N/A",
                "receiver": receiver if receiver else "Not provided",
                "content_length": len(content)
            },
            "plots": {
                "analysis": "/api/plots/svm_analysis.png",
                "comparison": "/api/plots/comprehensive_model_comparison.png"
            }
        })
    except Exception as e:
        logger.error(f"Error in SVM prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/predict/mlp', methods=['POST'])
def predict_mlp():
    try:
        data = request.get_json()
        if not data or 'message_type' not in data or 'content' not in data:
            return jsonify({"error": "Missing required fields: message_type and content"}), 400
        
        message_type = data.get('message_type', '')
        content = data.get('content', '')
        sender = data.get('sender', '') if message_type == 'email' else ''
        receiver = data.get('receiver', '') if message_type == 'email' else ''
        
        if message_type == 'email' and not sender:
            return jsonify({"error": "Sender email is required for email messages"}), 400
        
        result = mlp_service.predict(sender, content, receiver, message_type)
        if 'error' in result:
            return jsonify(result), 500
        return jsonify({
            "algorithm": "Multi-Layer Perceptron",
            "prediction": result,
            "input": {
                "message_type": message_type,
                "sender": sender if message_type == 'email' else "N/A",
                "receiver": receiver if receiver else "Not provided",
                "content_length": len(content)
            },
            "plots": {
                "analysis": "/api/plots/mlp_analysis.png",
                "comparison": "/api/plots/comprehensive_model_comparison.png"
            }
        })
    except Exception as e:
        logger.error(f"Error in MLP prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/predict/all', methods=['POST'])
def predict_all_algorithms():
    try:
        data = request.get_json()
        if not data or 'message_type' not in data or 'content' not in data:
            return jsonify({"error": "Missing required fields: message_type and content"}), 400
        
        message_type = data.get('message_type', '')
        content = data.get('content', '')
        sender = data.get('sender', '') if message_type == 'email' else ''
        receiver = data.get('receiver', '') if message_type == 'email' else ''
        
        if message_type == 'email' and not sender:
            return jsonify({"error": "Sender email is required for email messages"}), 400
        
        results = {
            "logistic_regression": lr_service.predict(sender, content, receiver, message_type),
            "random_forest": rf_service.predict(sender, content, receiver, message_type),
            "xgboost": xgb_service.predict(sender, content, receiver, message_type),
            "svm": svm_service.predict(sender, content, receiver, message_type),
            "mlp": mlp_service.predict(sender, content, receiver, message_type)
        }
        
        predictions = []
        confidences = []
        for result in results.values():
            if 'error' not in result:
                predictions.append(result['is_phishing'])
                confidences.append(result['confidence'])
        
        if predictions:
            ensemble_prediction = sum(predictions) > len(predictions) / 2
            ensemble_confidence = sum(confidences) / len(confidences) 
            
            prediction_data = {
                "timestamp": datetime.now().isoformat(),
                "input": {
                    "message_type": message_type,
                    "sender": sender if message_type == 'email' else "N/A",
                    "receiver": receiver if receiver else "Not provided",
                    "content_length": len(content)
                },
                "individual_predictions": results,
                "ensemble_prediction": {
                    "is_phishing": ensemble_prediction,
                    "confidence": ensemble_confidence,
                    "risk_level": "High" if ensemble_confidence > 0.7 else "Medium" if ensemble_confidence > 0.3 else "Low",
                    "algorithms_agreeing": sum(predictions),
                    "total_algorithms": len(predictions)
                },
                "plots": {
                    "logistic_regression": "/api/plots/logistic_regression_analysis.png",
                    "random_forest": "/api/plots/random_forest_analysis.png",
                    "xgboost": "/api/plots/xgboost_analysis.png",
                    "svm": "/api/plots/svm_analysis.png",
                    "mlp": "/api/plots/mlp_analysis.png",
                    "comparison": "/api/plots/comprehensive_model_comparison.png"
                }
            }
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(RESULTS_DIR, f"prediction_{timestamp}.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(prediction_data, f, indent=2, ensure_ascii=False)
            
            return jsonify(prediction_data)
        else:
            return jsonify({"error": "All algorithms failed to predict"}), 500
    except Exception as e:
        logger.error(f"Error in ensemble prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/plots/<filename>', methods=['GET'])
def get_plot(filename):
    """Serve plot files"""
    try:
        plot_path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(plot_path):
            return send_file(plot_path)
        else:
            return jsonify({"error": "Plot not found"}), 404
    except Exception as e:
        logger.error(f"Error serving plot {filename}: {e}")
        return jsonify({"error": "Failed to serve plot"}), 500

@app.route('/api/models/performance', methods=['GET'])
def get_models_performance():
    """Get performance metrics for all models"""
    try:
        performance_data = {}
        rf_csv = "data/random_forest_results.csv"
        if os.path.exists(rf_csv):
            import pandas as pd
            df = pd.read_csv(rf_csv)
            if not df.empty:
                performance_data['random_forest'] = df.to_dict('records')[-1]  
        xgb_csv = "data/xgboost_results.csv"
        if os.path.exists(xgb_csv):
            import pandas as pd
            df = pd.read_csv(xgb_csv)
            if not df.empty:
                performance_data['xgboost'] = df.to_dict('records')[-1]  
        
        return jsonify({"performance": performance_data})
    except Exception as e:
        logger.error(f"Error getting models performance: {e}")
        return jsonify({"error": "Failed to get models performance"}), 500

@app.route('/api/predict/enhanced', methods=['POST'])
def predict_enhanced():
    try:
        from utils.phishing_utils import PhishingUtils
        
        data = request.get_json()
        if not data or 'message_type' not in data or 'content' not in data:
            return jsonify({"error": "Missing required fields: message_type and content"}), 400
        
        message_type = data.get('message_type', '')
        content = data.get('content', '')
        sender = data.get('sender', '') if message_type == 'email' else ''
        model_type = data.get('model_type', 'logistic_regression')
        
        if message_type == 'email' and not sender:
            return jsonify({"error": "Sender email is required for email messages"}), 400
        
        model_service = lr_service
        if model_type == 'random_forest':
            model_service = rf_service
        elif model_type == 'xgboost':
            model_service = xgb_service
        elif model_type == 'svm':
            model_service = svm_service
        elif model_type == 'mlp':
            model_service = mlp_service
        
        result = PhishingUtils.enhanced_prediction_with_rules(sender, content, model_service, message_type)
        
        return jsonify({
            "algorithm": f"Enhanced {model_type.replace('_', ' ').title()}",
            "prediction": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in enhanced prediction: {e}")
        return jsonify({"error": f"Enhanced prediction failed: {str(e)}"}), 500

@app.route('/api/statistics/model/<model_name>', methods=['GET'])
def get_model_statistics(model_name):
    try:
        from utils.phishing_utils import PhishingUtils
        
        model_path = os.path.join('models', f'{model_name}_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model {model_name} not found"}), 404
        
        stats = PhishingUtils.get_model_statistics(model_path)
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting model statistics: {e}")
        return jsonify({"error": f"Failed to get model statistics: {str(e)}"}), 500

@app.route('/api/statistics/features', methods=['GET'])
def get_feature_statistics():
    try:
        from utils.phishing_utils import PhishingUtils
        
        stats = PhishingUtils.get_feature_distribution_comparison()
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting feature statistics: {e}")
        return jsonify({"error": f"Failed to get feature statistics: {str(e)}"}), 500

@app.route('/api/statistics/scaling/<path:url>', methods=['GET'])
def get_scaling_statistics(url):
    try:
        from utils.phishing_utils import PhishingUtils
        
        stats = PhishingUtils.get_scaling_analysis(url)
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting scaling statistics: {e}")
        return jsonify({"error": f"Failed to get scaling statistics: {str(e)}"}), 500

@app.errorhandler(404)
def not_found():
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error():
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 
