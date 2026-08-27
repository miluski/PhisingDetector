"""Balance dataset_full.csv / dataset_optimized.csv to 50/50 phishing/legitimate via random undersampling."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"
TARGET_COL = "phishing"
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("balance_url_dataset")


def balance_csv(input_path: Path, output_path: Path) -> None:
    """Undersample one CSV to equal binary-class counts."""
    df = pd.read_csv(input_path)
    before_counts = df[TARGET_COL].value_counts()
    logger.info("%s before: %s", input_path.name, before_counts.to_dict())

    sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    X_res, y_res = sampler.fit_resample(df.drop(columns=[TARGET_COL]), df[TARGET_COL])
    balanced = pd.concat([X_res, y_res], axis=1)
    balanced = balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    after_counts = balanced[TARGET_COL].value_counts()
    logger.info("%s after: %s", output_path.name, after_counts.to_dict())

    balanced.to_csv(output_path, index=False)
    logger.info("Wrote %s (%d rows)", output_path, len(balanced))


def main() -> None:
    """Balance full and optimized legacy datasets."""
    balance_csv(DATA_DIR / "dataset_full.csv", DATA_DIR / "dataset_full_balanced.csv")
    balance_csv(DATA_DIR / "dataset_optimized.csv", DATA_DIR / "dataset_optimized_balanced.csv")


if __name__ == "__main__":
    main()
