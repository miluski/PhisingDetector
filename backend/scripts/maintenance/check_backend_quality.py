"""Run repository-local structural quality checks for the Python backend."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    """One source-level quality finding."""

    path: Path
    line: int
    message: str

    def render(self) -> str:
        """Return a compiler-like finding string."""
        return f"{self.path}:{self.line}: {self.message}"


def python_files(root: Path, include_tests: bool = False):
    """Yield production Python files while skipping generated data and tests."""
    ignored = {"data", "models", "__pycache__"}
    if not include_tests:
        ignored.add("tests")
    return (path for path in root.rglob("*.py") if not ignored.intersection(path.parts))


def inspect_tree(path: Path, source: str) -> list[Finding]:
    """Collect AST findings for one source file."""
    tree = ast.parse(source, filename=str(path))
    findings = []
    findings.extend(find_long_definitions(path, tree))
    findings.extend(find_missing_docstrings(path, tree))
    findings.extend(find_swallowed_exceptions(path, tree))
    return findings


def find_long_definitions(path: Path, tree: ast.AST) -> list[Finding]:
    """Find functions and methods exceeding the project line budget."""
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = node.end_lineno - node.lineno + 1
            if length > 20:
                findings.append(Finding(path, node.lineno, f"{node.name}: {length} linii"))
    return findings


def find_missing_docstrings(path: Path, tree: ast.AST) -> list[Finding]:
    """Find public definitions without a docstring."""
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_") or ast.get_docstring(node):
            continue
        findings.append(Finding(path, node.lineno, f"{node.name}: brak docstringa"))
    return findings


def find_swallowed_exceptions(path: Path, tree: ast.AST) -> list[Finding]:
    """Find broad exception handlers whose only action is pass."""
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or not isinstance(node.type, ast.Name):
            continue
        if node.type.id == "Exception" and len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            findings.append(Finding(path, node.lineno, "except Exception: pass"))
    return findings


def find_long_lines(path: Path, source: str) -> list[Finding]:
    """Find physical lines longer than the configured 120-character limit."""
    return [Finding(path, number, "linia ma ponad 120 znaków")
            for number, line in enumerate(source.splitlines(), 1) if len(line) > 120]


def collect_findings(root: Path, include_tests: bool = False) -> list[Finding]:
    """Collect AST and physical-line findings for the selected backend tree."""
    findings = []
    for path in python_files(root, include_tests):
        source = path.read_text(encoding="utf-8")
        findings.extend(inspect_tree(path, source))
        findings.extend(find_long_lines(path, source))
    return sorted(findings, key=lambda item: (str(item.path), item.line, item.message))


def parse_args() -> argparse.Namespace:
    """Parse quality-check command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--strict", action="store_true", help="zwróć kod 1, gdy znaleziono problemy")
    parser.add_argument("--include-tests", action="store_true", help="uwzględnij moduły testowe w audycie")
    return parser.parse_args()


def main() -> int:
    """Run checks, print findings and return a shell status code."""
    args = parse_args()
    findings = collect_findings(args.root, args.include_tests)
    for finding in findings:
        print(finding.render())
    print(f"Znaleziono: {len(findings)}")
    return int(bool(findings) and args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
