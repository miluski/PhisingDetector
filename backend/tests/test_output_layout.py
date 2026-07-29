from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent


def test_data_root_contains_only_datasets_and_test_inputs():
    data_root = BACKEND_ROOT / "data"
    assert {path.name for path in data_root.iterdir()} == {
        "dataset_full.csv",
        "dataset_full_balanced.csv",
        "dataset_optimized.csv",
        "dataset_optimized_balanced.csv",
        "test",
        "url_corpus",
    }
    assert not (data_root / "phase_testing_results").exists()
    assert not (data_root / "url_corpus" / "dataset_optimized.csv").exists()


def test_documentation_has_no_old_result_paths():
    markdown = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]
    )
    assert "backend/data/phase_testing_results" not in markdown
    assert "backend/data/mlp_comparison.csv" not in markdown
    assert "backend/data/url_corpus/dataset_optimized.csv" not in markdown
