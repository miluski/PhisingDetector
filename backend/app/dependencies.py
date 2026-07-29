"""Application composition root and dependency accessors."""

from __future__ import annotations

from dataclasses import dataclass

from services.phishing_detection_service import PhishingDetectionService
from services.model_accuracy_service import ModelAccuracyService
from services.message_example_service import MessageExampleService
from services.url_feature_service import URLFeatureService
from validators.dataset_validator import DatasetValidator
from validators.mode_validator import ModeValidator
from validators.model_validator import ModelValidator
from validators.url_validator import URLValidator


@dataclass(frozen=True)
class ApplicationDependencies:
    """Concrete services used by the HTTP resources."""

    feature_service: URLFeatureService
    phishing_service: PhishingDetectionService
    url_validator: URLValidator
    mode_validator: ModeValidator
    dataset_validator: DatasetValidator
    model_validator: ModelValidator
    model_accuracy_service: ModelAccuracyService
    message_example_service: MessageExampleService


def build_dependencies() -> ApplicationDependencies:
    """Construct production dependencies in one composition-root function."""
    return ApplicationDependencies(
        feature_service=URLFeatureService(),
        phishing_service=PhishingDetectionService(),
        url_validator=URLValidator(),
        mode_validator=ModeValidator(),
        dataset_validator=DatasetValidator(),
        model_validator=ModelValidator(),
        model_accuracy_service=ModelAccuracyService(),
        message_example_service=MessageExampleService(),
    )


dependencies = build_dependencies()
feature_service = dependencies.feature_service
phishing_service = dependencies.phishing_service
url_validator = dependencies.url_validator
mode_validator = dependencies.mode_validator
dataset_validator = dependencies.dataset_validator
model_validator = dependencies.model_validator
model_accuracy_service = dependencies.model_accuracy_service
message_example_service = dependencies.message_example_service
