import pytest

from backend.scripts.data_pipeline.audit_benign_category_coverage import validate


def _write_split(path, rows):
    categories = [
        "about", "contact", "docs", "help", "playlist", "pricing", "status",
        "watch", "products", "login", "account", "secure", "support", "payment", "verify",
    ]
    path.write_text(
        "url,label\n" + "\n".join(
            [f"https://trusted.example/{category},0" for category in categories] + rows
        ) + "\n",
        encoding="utf-8",
    )


def test_category_audit_accepts_complete_split(tmp_path):
    path = tmp_path / "train.csv"
    _write_split(path, ["https://phish.example/login,1"])

    report = validate([path], minimum=1)

    assert report["passed"] is True


def test_category_audit_rejects_underrepresented_split(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("url,label\nhttps://trusted.example/docs,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="category_coverage_failures"):
        validate([path], minimum=1)
