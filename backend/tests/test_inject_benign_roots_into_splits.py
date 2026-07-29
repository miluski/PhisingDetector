import csv

from scripts.inject_benign_roots_into_splits import inject


HEADER = [
    "source_index", "url", "label", "registered_domain", "host",
    "template_key", "normalized_path", "query_keys",
]


def write_csv(path, rows, header=HEADER):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def test_injection_preserves_balance_and_domain_isolation(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    roots = tmp_path / "roots.csv"
    index = 0
    for split in ("train", "val", "test"):
        rows = []
        for label in (0, 1):
            for number in range(2):
                domain = f"{split}-{label}-{number}.com"
                rows.append((index, f"https://{domain}/path", label, domain, domain, str(index), "/path", ""))
                index += 1
        write_csv(source / f"raw_{split}.csv", rows)
    write_csv(
        roots,
        [(f"https://root-{number}.com/", 0) for number in range(6)],
        header=["url", "label"],
    )

    report = inject(source, roots, output, root_limit=6, seed=42)

    assert report["selected_roots"] == 6
    assert report["domain_overlap"] == {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }
    for split in ("train", "val", "test"):
        assert report["splits"][split]["rows"] == 4
        assert report["splits"][split]["benign"] == 2
        assert report["splits"][split]["phishing"] == 2
