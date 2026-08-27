"""Remove obsolete archives and checksum-identical notebook outputs."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


@dataclass(frozen=True)
class CleanupCandidate:
    """One allow-listed cleanup target."""

    path: Path
    kind: str


class ArtifactCleaner:
    """Find and remove obsolete or checksum-identical artifacts."""

    def __init__(self, backend_root: Path, results_root: Path):
        """Create cleaner with explicit backend and result roots."""
        self.backend_root = backend_root
        self.results_root = results_root

    def candidates(self) -> list[CleanupCandidate]:
        """Return every safe cleanup candidate."""
        return [*self._archives(), *self._duplicate_notebooks()]

    def _archives(self) -> list[CleanupCandidate]:
        archive_root = self.backend_root / "archive"
        paths = sorted(archive_root.glob("project_versions_*"))
        return [CleanupCandidate(path, "obsolete archive") for path in paths if path.is_dir()]

    def _duplicate_notebooks(self) -> list[CleanupCandidate]:
        source_by_name = self._source_notebooks()
        output_root = self.results_root / "notebook_runs"
        outputs = sorted(output_root.glob("*.ipynb"))
        return [CleanupCandidate(path, "duplicate notebook") for path in outputs
                if self._matches_source(path, source_by_name)]

    def _source_notebooks(self) -> dict[str, Path]:
        notebook_root = self.backend_root / "notebooks"
        return {path.name: path for path in notebook_root.rglob("*.ipynb")}

    def _matches_source(self, output: Path, sources: dict[str, Path]) -> bool:
        source = sources.get(output.name)
        return source is not None and self._digest(source) == self._digest(output)

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def run(self, apply: bool) -> list[CleanupCandidate]:
        """Preview candidates or remove them when apply is enabled."""
        candidates = self.candidates()
        for candidate in candidates:
            print(f"{'DELETE' if apply else 'WOULD DELETE'} [{candidate.kind}]: {candidate.path}")
            if apply:
                self._remove(candidate.path)
        if apply:
            self._remove_empty_directories()
        return candidates

    @staticmethod
    def _remove(path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def _remove_empty_directories(self) -> None:
        roots = [self.backend_root / "archive", self.results_root / "notebook_runs"]
        for root in roots:
            self._prune(root)

    @staticmethod
    def _prune(root: Path) -> None:
        if not root.exists():
            return
        directories = sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True)
        for directory in directories:
            if not any(directory.iterdir()):
                directory.rmdir()
        if not any(root.iterdir()):
            root.rmdir()


def parse_args() -> argparse.Namespace:
    """Parse explicit deletion flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run artifact cleanup from canonical project roots."""
    args = parse_args()
    cleaner = ArtifactCleaner(BACKEND_ROOT, PROJECT_ROOT / "results")
    candidates = cleaner.run(args.apply)
    print(f"Candidates: {len(candidates)}")


if __name__ == "__main__":
    main()
