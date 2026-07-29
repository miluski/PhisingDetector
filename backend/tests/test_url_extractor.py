from extractors.content.url_extractor import URLExtractor


def test_literal_escaped_newlines_terminate_url():
    text = (
        r"https://verify.com "
        r"https://gmail.com "
        r"https://bankofamerica-verify.com/secure/login?id=12345&token=abc\n\nIf "
        r"https://bankofamerica-verify.com"
    )

    assert URLExtractor().extract_from_text(text) == {
        "https://verify.com",
        "https://gmail.com",
        "https://bankofamerica-verify.com/secure/login?id=12345&token=abc",
        "https://bankofamerica-verify.com",
    }


def test_backslash_cannot_be_part_of_extracted_url():
    urls = URLExtractor().extract_from_text(
        r"Open https://example.com/path\\unexpected-text"
    )

    assert urls == {"https://example.com/path"}


def test_bare_domain_is_not_extracted_from_inside_full_url():
    text = "Secure your account at https://paypal-security-center.net/verify/account"

    assert URLExtractor().extract_from_text(text) == {
        "https://paypal-security-center.net/verify/account"
    }


def test_support_tld_extracts_phishing_link_and_email_domains():
    extractor = URLExtractor()
    body = "Unlock: https://apple-id-verify.support/unlock?appleid=iphoneuser"

    assert extractor.extract_from_text(body) == {
        "https://apple-id-verify.support/unlock?appleid=iphoneuser"
    }
    assert set(extractor.extract_all_urls(
        "appleid@apple-id-verify.support", "iphoneuser@icloud.com", body
    )) == {
        "https://apple-id-verify.support/unlock?appleid=iphoneuser",
        "https://apple-id-verify.support",
        "https://icloud.com",
    }


def test_extracts_domains_with_tlds_outside_legacy_whitelist():
    extractor = URLExtractor()

    assert set(extractor.extract_all_urls(
        "alerts@account-security.security",
        "user@new-mail.example",
        "Visit https://login.account-security.security/confirm",
    )) == {
        "https://account-security.security",
        "https://new-mail.example",
        "https://login.account-security.security/confirm",
    }
