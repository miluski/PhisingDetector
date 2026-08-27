from backend.scripts.data_pipeline.build_quality_url_corpus import (
    build_hard_negative_rows,
    exclude_benign_roots_on_phishing_domains,
    is_benign_root,
    registered_domain,
    sample_balanced,
)


def test_benign_root_detection():
    assert is_benign_root("https://gmail.com/")
    assert is_benign_root("https://example.org")
    assert not is_benign_root("https://example.org/login")
    assert not is_benign_root("https://example.org/?next=login")
    assert registered_domain("https://www.gov.uk/") == "gov.uk"


def test_balanced_sample_guarantees_requested_benign_roots():
    rows = [
        ("https://gmail.com/", 0),
        ("https://google.com/", 0),
        ("https://example.org/docs", 0),
        ("https://safe.example.net/help", 0),
        ("https://evil.test/login-a", 1),
        ("https://evil.test/login-b", 1),
        ("https://phish.test/login-a", 1),
        ("https://phish.test/login-b", 1),
    ]

    sampled = sample_balanced(rows, per_class_limit=4, seed=42, minimum_benign_roots=2)

    assert sum(label == 0 for _, label in sampled) == 4
    assert sum(label == 1 for _, label in sampled) == 4
    assert sum(label == 0 and is_benign_root(url) for url, label in sampled) >= 2


def test_phishing_domain_collision_removes_only_benign_root():
    rows = [
        ("https://shared.example/", 0),
        ("https://shared.example/documentation", 0),
        ("https://shared.example/login", 1),
        ("https://gmail.com/", 0),
    ]

    filtered, removed = exclude_benign_roots_on_phishing_domains(rows)

    assert removed == 1
    assert ("https://shared.example/", 0) not in filtered
    assert ("https://shared.example/documentation", 0) in filtered
    assert ("https://gmail.com/", 0) in filtered


def test_hard_negative_generator_avoids_phishing_domains_and_adds_sensitive_paths():
    rows = build_hard_negative_rows(
        ["example.com", "evil.example"],
        {"evil.example"},
        set(),
        per_domain=6,
    )

    assert rows
    assert all(label == 0 for _, label in rows)
    assert all("evil.example" not in url for url, _ in rows)
    assert any(token in url for url, _ in rows for token in ("login", "payment", "verify"))
