"""Archive obsolete generated dataset and model versions without data loss."""

from __future__ import annotations

import argparse
import shutil
from datetime import date
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = BACKEND_ROOT / "data" / "url_corpus"
MODEL_ROOT = BACKEND_ROOT / "models"
ARCHIVE_ROOT = BACKEND_ROOT / "archive" / f"project_versions_{date.today().isoformat()}"
OLD_DATA = (
    "quality_v2_features", "quality_v2_splits", "quality_v3_features",
    "quality_large_splits", "quality_large_splits_pre_benign_roots",
    "quality_corrected_features", "quality_corrected_splits",
    "raw_urls_quality.csv", "raw_urls_quality_v2.csv", "raw_urls_quality_corrected.csv",
)
OLD_MODELS = ("staged_url_corpus_v2", "staged_url_corpus_v3", "staged_url_corpus_corrected")


def parse_args() -> argparse.Namespace:
    """Parse cleanup mode and optional model archival flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Move candidates into the archive")
    parser.add_argument("--include-models", action="store_true", help="Also archive old staged models")
    return parser.parse_args()


def candidates(include_models: bool) -> list[Path]:
    """Return explicitly allow-listed obsolete project artifacts."""
    paths = [DATA_ROOT / name for name in OLD_DATA]
    if include_models:
        paths.extend(MODEL_ROOT / name for name in OLD_MODELS)
    return [path for path in paths if path.exists()]


def stale_temporary_files() -> list[Path]:
    """Return completed temporary markers whose final CSV already exists."""
    result = []
    for marker in DATA_ROOT.rglob("*.tmp.done"):
        final = Path(str(marker).removesuffix(".tmp.done"))
        if final.exists() and final.stat().st_mtime >= marker.stat().st_mtime:
            result.append(marker)
    return result


def archive_path(path: Path) -> Path:
    """Map a project path to an equivalent location below the archive."""
    relative = path.relative_to(BACKEND_ROOT)
    return ARCHIVE_ROOT / relative


def archive(paths: list[Path], apply: bool) -> None:
    """Print candidates and optionally move them into the dated archive."""
    all_paths = [*paths, *stale_temporary_files()]
    if not all_paths:
        print("Brak bezpiecznych artefaktów do archiwizacji.")
        return
    for path in all_paths:
        target = archive_path(path)
        print(f"{'ARCHIWIZUJĘ' if apply else 'DO ARCHIWIZACJI'}: {path} -> {target}")
        if apply and path.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                print(f"POMIJAM — cel już istnieje: {target}")
                continue
            shutil.move(str(path), str(target))


def main() -> None:
    """Run a dry-run by default; archive only with explicit ``--apply``."""
    args = parse_args()
    archive(candidates(args.include_models), args.apply)


if __name__ == "__main__":
    main()
