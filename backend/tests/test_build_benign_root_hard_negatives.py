import csv

from backend.scripts.data_pipeline.build_benign_root_hard_negatives import build


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["url", "label"])
        writer.writerows(rows)


def test_build_keeps_unique_roots_and_rejects_phishing_domains(tmp_path):
    ranked = tmp_path / "ranked.csv"
    threats = tmp_path / "threats.csv"
    output = tmp_path / "roots.csv"
    write_rows(ranked, [
        ("https://gmail.com/", 0),
        ("https://shared.example/", 0),
        ("https://gmail.com/path", 0),
        ("https://evil.example/", 1),
    ])
    write_rows(threats, [("https://shared.example/login", 1)])

    stats = build([ranked], [threats], output, limit=None)

    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows == [{"url": "https://gmail.com/", "label": "0"}]
    assert stats["written"] == 1
    assert stats["rejected_phishing_domain"] == 1
