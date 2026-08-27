"""Report unexpected checksum-identical tracked files."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_GROUPS = {
    frozenset({
        "backend/config/__init__.py",
        "backend/extractors/__init__.py",
        "backend/extractors/lexical/__init__.py",
        "backend/extractors/network/__init__.py",
        "backend/services/__init__.py",
        "backend/validators/__init__.py",
    }),
}


def tracked_files(root: Path) -> list[Path]:
    """Return tracked and visible untracked files in repository."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root, check=True, capture_output=True,
    )
    names = result.stdout.decode().split("\0")
    return [root / name for name in names if name and (root / name).is_file()]


def digest(path: Path) -> str:
    """Return SHA-256 digest for one file."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def duplicate_groups(root: Path) -> list[frozenset[str]]:
    """Group repository files sharing size and digest."""
    groups: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for path in tracked_files(root):
        groups[(path.stat().st_size, digest(path))].append(path)
    return [frozenset(str(path.relative_to(root)) for path in paths)
            for paths in groups.values() if len(paths) > 1]


def unexpected_duplicates(root: Path) -> list[frozenset[str]]:
    """Return duplicate groups absent from semantic allowlist."""
    return [group for group in duplicate_groups(root) if group not in ALLOWED_GROUPS]


def parse_args() -> argparse.Namespace:
    """Parse optional repository root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> None:
    """Print unexpected groups and fail when any exist."""
    groups = unexpected_duplicates(parse_args().root.resolve())
    for group in groups:
        print("\n".join(sorted(group)))
        print()
    raise SystemExit(1 if groups else 0)


if __name__ == "__main__":
    main()
