from utils.benign_url_categories import classify_benign_category, is_calibration_url


def test_classifies_generic_and_sensitive_benign_path_categories():
    assert classify_benign_category("https://trusted.example/docs") == "docs"
    assert classify_benign_category("https://trusted.example/playlist/example") == "playlist"
    assert classify_benign_category("https://trusted.example/login") == "login"
    assert classify_benign_category("https://trusted.example/verify-email") == "verify"


def test_root_and_unknown_paths_are_not_calibration_categories():
    assert classify_benign_category("https://trusted.example/") is None
    assert not is_calibration_url("https://trusted.example/unclassified")
