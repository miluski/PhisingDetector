"""Generate final inter-model comparisons without rerunning model training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
os.environ.setdefault("MPLBACKEND", "Agg")
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from utils.final_model_comparison import generate_final_model_comparisons


def parse_args() -> argparse.Namespace:
    """Parse result scope and optional project result directory."""
    parser = argparse.ArgumentParser(
        description="Buduje końcowe porównania modeli z zapisanych plików CSV."
    )
    parser.add_argument(
        "--scope", choices=("original", "url_corpus", "all"), default="all",
    )
    parser.add_argument(
        "--results-root", type=Path, default=PROJECT_ROOT / "results",
    )
    return parser.parse_args()


def selected_scopes(scope: str) -> tuple[str, ...]:
    """Return concrete result scopes requested by the caller."""
    if scope == "all":
        return "original", "url_corpus"
    return (scope,)


def build_scope(results_root: Path, scope: str) -> list[Path]:
    """Generate one scope and return its localized Polish plot paths."""
    root = results_root / scope
    frame, paths = generate_final_model_comparisons(
        root / "model_comparisons", root / "final_model_comparison"
    )
    print(f"{scope}: {len(frame)} wierszy, {len(paths)} wykresów")
    return paths


def main() -> int:
    """Generate comparisons for all requested completed result scopes."""
    args = parse_args()
    for scope in selected_scopes(args.scope):
        paths = build_scope(args.results_root, scope)
        for path in paths:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
