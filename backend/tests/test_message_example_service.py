"""Tests for the backend message example pool."""

import json

from services.message_example_service import MessageExampleService


def test_message_example_service_combines_email_and_sms_groups(tmp_path):
    source = tmp_path / "examples.json"
    source.write_text(
        json.dumps(
            {
                "phishing_emails": [{"body": "phishing", "sender": "a", "recipient": "b", "expected_result": "phishing"}],
                "legitimate_emails": [{"body": "safe", "sender": "c", "recipient": "d", "expected_result": "safe"}],
                "phishing_sms": [{"body": "phishing sms", "expected_result": "phishing"}],
                "legitimate_sms": [{"body": "safe sms", "expected_result": "safe"}],
            }
        ),
        encoding="utf-8",
    )

    examples = MessageExampleService(source).get_examples()

    assert [item["body"] for item in examples["emails"]] == ["phishing", "safe"]
    assert [item["body"] for item in examples["sms"]] == ["phishing sms", "safe sms"]
    assert examples["emails"][0]["contains_phishing_url"] is True
    assert examples["sms"][1]["contains_phishing_url"] is False
