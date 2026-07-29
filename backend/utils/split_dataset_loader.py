from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


METADATA_COLUMNS = {"source_index", "url", "missing_whois"}
LABEL_COLUMN = "phishing"


def build_stratified_frames(
    dataframe: pd.DataFrame,
    *,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Create deterministic train/validation/test frames preserving class ratio."""
    _validate_split_request(dataframe, validation_size, test_size)
    train_validation, test = _split_test(dataframe, test_size, random_state)
    train, validation = _split_validation(train_validation, validation_size, test_size, random_state)
    return {"train": train.copy(), "val": validation.copy(), "test": test.copy()}


def _validate_split_request(dataframe, validation_size, test_size):
    """Validate a binary-labelled split request."""
    if LABEL_COLUMN not in dataframe.columns:
        raise ValueError(f"Missing label column: {LABEL_COLUMN}")
    if validation_size < 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be in [0, 1)")


def _split_test(dataframe, test_size, random_state):
    """Separate the held-out test frame with stratification."""
    return train_test_split(
        dataframe, test_size=test_size, random_state=random_state,
        stratify=dataframe[LABEL_COLUMN]
    )


def _split_validation(train_validation, validation_size, test_size, random_state):
    """Separate validation rows or return an empty validation frame."""
    if validation_size == 0:
        return train_validation, train_validation.iloc[0:0].copy()
    relative_size = validation_size / (1.0 - test_size)
    return train_test_split(
        train_validation, test_size=relative_size, random_state=random_state,
        stratify=train_validation[LABEL_COLUMN]
    )


def build_controlled_imbalance_frame(
    dataframe: pd.DataFrame,
    *,
    majority_label: int = 0,
    majority_to_minority_ratio: float = 58000 / 30647,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a deterministic binary-class subset with a requested imbalance."""
    _validate_imbalance_request(dataframe, majority_to_minority_ratio)
    majority, minority = _class_frames(dataframe, majority_label)
    minority_count = _minority_count(majority, minority, majority_to_minority_ratio)
    majority_count = min(len(majority), int(round(minority_count * majority_to_minority_ratio)))
    return _sample_imbalanced(majority, minority, majority_count, minority_count, random_state)


def _validate_imbalance_request(dataframe, ratio):
    """Validate label and ratio assumptions for controlled sampling."""
    if LABEL_COLUMN not in dataframe.columns:
        raise ValueError(f"Missing label column: {LABEL_COLUMN}")
    if ratio <= 1:
        raise ValueError("majority_to_minority_ratio must be greater than 1")
    labels = set(dataframe[LABEL_COLUMN].dropna().unique())
    if labels != {0, 1}:
        raise ValueError(f"Expected binary labels {{0, 1}}, got {labels}")


def _class_frames(dataframe, majority_label):
    """Return majority and minority class frames."""
    return (
        dataframe[dataframe[LABEL_COLUMN] == majority_label],
        dataframe[dataframe[LABEL_COLUMN] == 1 - majority_label],
    )


def _minority_count(majority, minority, ratio):
    """Compute the minority sample count under the requested ratio."""
    count = min(len(minority), int(len(majority) / ratio))
    if count == 0:
        raise ValueError("Not enough rows to construct controlled imbalance")
    return count


def _sample_imbalanced(majority, minority, majority_count, minority_count, random_state):
    """Sample and shuffle the requested class counts."""
    sampled = pd.concat([
        majority.sample(n=majority_count, random_state=random_state),
        minority.sample(n=minority_count, random_state=random_state),
    ], ignore_index=True)
    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)


def load_split_frames(split_dir: str | Path, dataset_prefix: str = "dataset_full") -> Dict[str, pd.DataFrame]:
    """Load predefined train, validation and test CSV frames."""
    split_dir = Path(split_dir)
    paths = {
        "train": split_dir / f"{dataset_prefix}_train.csv",
        "val": split_dir / f"{dataset_prefix}_val.csv",
        "test": split_dir / f"{dataset_prefix}_test.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing predefined split dataset files. Generate feature datasets "
            "for train/val/test before training. Missing: " + ", ".join(missing)
        )
    return {name: pd.read_csv(path) for name, path in paths.items()}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model features after removing labels and metadata columns."""
    return [
        column for column in df.columns
        if column not in METADATA_COLUMNS and column != LABEL_COLUMN
    ]


def _scaled_frames(frames, columns):
    """Extract feature frames and fit a scaler on training data only."""
    values = [frames[name][columns] for name in ("train", "val", "test")]
    scaler = StandardScaler()
    scaled = [scaler.fit_transform(values[0])]
    scaled.extend(scaler.transform(frame) for frame in values[1:])
    return values, scaled, scaler


def _train_test_payload(frames):
    """Build the model-ready dictionary for one dataset name."""
    train_df = frames["train"]
    columns = feature_columns(train_df)
    values, scaled, scaler = _scaled_frames(frames, columns)
    return {
        "X_train": values[0], "y_train": train_df[LABEL_COLUMN],
        "X_val": values[1], "y_val": frames["val"][LABEL_COLUMN],
        "X_test": values[2], "y_test": frames["test"][LABEL_COLUMN],
        "X_train_scaled": scaled[0], "X_val_scaled": scaled[1], "X_test_scaled": scaled[2],
        "scaler": scaler, "feature_count": len(columns), "feature_columns": columns,
    }


def build_train_test_data(split_frames_by_name: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, dict]:
    """Convert named split frames into scaled model-ready dictionaries."""
    return {name: _train_test_payload(frames) for name, frames in split_frames_by_name.items()}


def print_split_summary(split_frames_by_name: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Print row, feature and class counts for predefined splits."""
    print("=" * 80)
    print("DANE ZAŁADOWANE - PREDEFINED DOMAIN/TEMPLATE SPLITS")
    print("=" * 80)
    for name, frames in split_frames_by_name.items():
        print(f"\n{name}:")
        for split_name, df in frames.items():
            label_counts = df[LABEL_COLUMN].value_counts().to_dict()
            safe = label_counts.get(0, 0)
            phishing = label_counts.get(1, 0)
            print(
                f"  • {split_name}: {df.shape[0]:,} rows, "
                f"{len(feature_columns(df))} features, "
                f"safe={safe:,}, phishing={phishing:,}"
            )
    _print_balance_notice()
    print("\nNo random train_test_split is used for URL corpus notebooks.")
    print("=" * 80)


def _print_balance_notice() -> None:
    """Explain the controlled class balance used by URL-corpus experiments."""
    print(
        "\nUwaga: zbiory URL-corpus są celowo zbalansowane (50% bezpiecznych, "
        "50% phishingowych) na potrzeby kontrolowanego porównania algorytmów."
    )
    print(
        "Wyników precision i accuracy nie należy bezpośrednio interpretować "
        "jako częstości błędów w ruchu produkcyjnym."
    )
