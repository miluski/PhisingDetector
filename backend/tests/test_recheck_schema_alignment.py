"""Regression tests for external recheck feature alignment."""

import pickle

import pandas as pd

from backend.scripts.modeling.recheck_url_corpus_models import (
    align_features,
    cached_features_are_compatible,
    load_payload,
    load_optional_frame,
    production_artifact_filename,
)


def test_missing_external_features_use_shared_missing_value():
    """External extraction may omit network fields without aborting recheck."""
    aligned = align_features(pd.DataFrame({"length": [10]}), ["length", "domain_in_ip"])
    assert aligned.columns.tolist() == ["length", "domain_in_ip"]
    assert aligned.loc[0, "domain_in_ip"] == -1


def test_cache_rejects_missing_deterministic_domain_feature():
    frame = pd.DataFrame({"domain_in_ip": [-1, -1], "length_url": [10, 20]})
    payloads = {"model": {"feature_names": ["domain_in_ip", "length_url"]}}

    assert not cached_features_are_compatible(frame, payloads)


def test_stale_smaller_cache_is_projected_from_valid_full_cache(tmp_path):
    cache_dir = tmp_path / "external_features"
    cache_dir.mkdir()
    full = pd.DataFrame({
        "url": ["https://example.com", "http://192.0.2.10"],
        "source_index": [1, 2],
        "length_url": [19, 17],
        "domain_in_ip": [0, 1],
        "phishing": [0, 1],
    })
    full.to_csv(cache_dir / "external_full.csv", index=False)
    stale = full.assign(domain_in_ip=-1)
    stale.to_csv(cache_dir / "external_optimized.csv", index=False)
    payloads = {"model": {"feature_names": ["domain_in_ip", "length_url"]}}

    result = load_optional_frame(
        tmp_path / "unused.csv", cache_dir, "optimized", payloads, True
    )

    assert result["domain_in_ip"].tolist() == [0, 1]
    persisted = pd.read_csv(cache_dir / "external_optimized.csv")
    assert persisted["domain_in_ip"].tolist() == [0, 1]


def test_recheck_loads_the_payload_written_by_model_notebooks(tmp_path):
    """Production recheck should accept the notebook artifact naming scheme."""
    model_dir = tmp_path / "random_forest"
    model_dir.mkdir()
    path = model_dir / production_artifact_filename("random_forest", "optimized")
    with path.open("wb") as file:
        pickle.dump({"model": "model", "decision_threshold": 0.7}, file)

    payload = load_payload(tmp_path, "random_forest", "optimized")

    assert payload["decision_threshold"] == 0.7
