from services.url_feature_service import URLFeatureService
from services.phishing_detection_service import PhishingDetectionService
from validators.url_validator import URLValidator
from validators.mode_validator import ModeValidator
from validators.dataset_validator import DatasetValidator

feature_service = URLFeatureService()
phishing_service = PhishingDetectionService()
url_validator = URLValidator()
mode_validator = ModeValidator()
dataset_validator = DatasetValidator()
