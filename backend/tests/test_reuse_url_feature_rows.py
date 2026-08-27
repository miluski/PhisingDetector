import csv

from backend.scripts.data_pipeline.reuse_url_feature_rows import reuse


RAW_HEADER = ["source_index", "url", "label"]
FEATURE_HEADER = ["source_index", "url", "length_url", "phishing"]


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def test_reuses_features_across_old_split_boundaries(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "output"
    write_csv(raw_dir / "raw_train.csv", RAW_HEADER, [(10, "https://a.test/", 0)])
    write_csv(raw_dir / "raw_val.csv", RAW_HEADER, [(11, "https://b.test/", 1)])
    write_csv(raw_dir / "raw_test.csv", RAW_HEADER, [(12, "https://new.test/", 0)])
    old_train = tmp_path / "old_train.csv"
    old_test = tmp_path / "old_test.csv"
    write_csv(old_train, FEATURE_HEADER, [(1, "https://b.test/", 20, 1)])
    write_csv(old_test, FEATURE_HEADER, [(2, "https://a.test/", 10, 0)])

    report = reuse(raw_dir, [old_train, old_test], output_dir)

    assert report["train"] == {"raw": 1, "reused": 1, "missing": 0}
    assert report["val"] == {"raw": 1, "reused": 1, "missing": 0}
    assert report["test"] == {"raw": 1, "reused": 0, "missing": 1}
    with (output_dir / "dataset_full_train_reused.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        row = next(csv.DictReader(file))
    assert row["source_index"] == "10"
    assert row["url"] == "https://a.test/"
