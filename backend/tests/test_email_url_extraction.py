from services.phishing_detection_service import PhishingDetectionService


class _OperationLogger:
    def step(self, *args, **kwargs):
        pass

    def success(self, *args, **kwargs):
        pass


def test_email_url_extraction_includes_receiver_domain_without_url_fragments():
    service = PhishingDetectionService.__new__(PhishingDetectionService)
    service.url_extractor = __import__(
        "extractors.content.url_extractor", fromlist=["URLExtractor"]
    ).URLExtractor()

    urls = service._extract_email_urls(
        _OperationLogger(),
        "service@paypal-security-center.net",
        "victim@gmail.com",
        "Secure your account: https://paypal-security-center.net/verify/account",
        None,
    )

    assert set(urls) == {
        "https://paypal-security-center.net",
        "https://paypal-security-center.net/verify/account",
        "https://gmail.com",
    }


def test_receiver_domain_is_analyzed_by_models_and_affects_verdict():
    service = PhishingDetectionService.__new__(PhishingDetectionService)
    service.visualizer = type(
        "Visualizer", (), {"create_comprehensive_report": lambda *_args, **_kwargs: {}}
    )()

    result = service._create_success_result(
        {
            "url": "https://gmail.com",
            "predictions": [],
            "ensemble": {
                "ensemble_prediction": 1,
                "ensemble_label": "Phishing",
            },
            "features": {},
            "similar_samples": [],
        },
        receiver_domain=True,
    )

    assert result["raw_is_phishing"] is True
    assert result["is_phishing"] is True
    assert result["receiver_domain"] is True
    assert "analizowana przez modele" in result["classification_note"]
