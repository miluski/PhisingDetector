from flask import Flask
from flask_cors import CORS
from config.logging_config import setup_logging, get_logger

setup_logging("DEBUG")
logger = get_logger("App")

logger.info("=" * 60)
logger.info("🚀 PHISHING DETECTOR BACKEND STARTING")
logger.info("=" * 60)

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": ["https://localhost:4000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

logger.info("✅ Flask app initialized with CORS")
