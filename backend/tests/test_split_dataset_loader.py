import pandas as pd

from utils.split_dataset_loader import (
    build_controlled_imbalance_frame,
    build_stratified_frames,
    feature_columns,
)


def test_stratified_frames_are_disjoint_and_preserve_rows():
    dataframe = pd.DataFrame(
        {
            "source_index": range(100),
            "url": [f"https://example-{index}.test" for index in range(100)],
            "feature": range(100),
            "phishing": [0] * 60 + [1] * 40,
        }
    )

    frames = build_stratified_frames(dataframe)

    assert {name: len(frame) for name, frame in frames.items()} == {
        "train": 60,
        "val": 20,
        "test": 20,
    }
    assert sum(len(frame) for frame in frames.values()) == len(dataframe)
    assert all(frame["phishing"].mean() == 0.4 for frame in frames.values())


def test_feature_columns_remove_metadata_and_label():
    dataframe = pd.DataFrame(
        {
            "source_index": [1],
            "url": ["https://example.test"],
            "feature": [2.0],
            "phishing": [0],
        }
    )

    assert feature_columns(dataframe) == ["feature"]


def test_controlled_imbalance_is_deterministic_and_preserves_schema():
    dataframe = pd.DataFrame(
        {
            "source_index": range(200),
            "url": [f"https://example-{index}.test" for index in range(200)],
            "feature": range(200),
            "phishing": [0] * 100 + [1] * 100,
        }
    )

    first = build_controlled_imbalance_frame(
        dataframe, majority_to_minority_ratio=2.0, random_state=7
    )
    second = build_controlled_imbalance_frame(
        dataframe, majority_to_minority_ratio=2.0, random_state=7
    )

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == list(dataframe.columns)
    assert first["phishing"].value_counts().to_dict() == {0: 100, 1: 50}
