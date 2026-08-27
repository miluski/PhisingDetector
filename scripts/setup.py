"""Validate project artifacts, prepare certificates, and start Docker services."""

from __future__ import annotations

import argparse
import os
import ssl
import subprocess
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
CERTIFICATE_ROOT = PROJECT_ROOT / "certs" / "localhost"
MODEL_STEMS = {
    "logistic_regression": "logistic_regression",
    "mlp": "mlp_model",
    "random_forest": "random_forest",
    "svm": "svm_model",
    "xgboost": "xgboost_model",
}
SCALER_MODELS = ("logistic_regression", "mlp", "svm")
MODES = ("full", "optimized")
DATASETS = ("current", "url_corpus")
SERVICE_URLS = ("https://localhost:5001/api/health", "https://localhost:4000")
CommandRunner = Callable[[Sequence[str], bool, bool], subprocess.CompletedProcess[str]]


class SetupError(RuntimeError):
    """Report actionable setup failure."""

    pass


def run_command(
    command: Sequence[str], check: bool = True, capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one project command with canonical environment."""
    return subprocess.run(
        list(command), cwd=PROJECT_ROOT, env=command_environment(), check=check,
        capture_output=capture, text=True,
    )


def command_environment() -> dict[str, str]:
    """Return Compose environment rooted at current project."""
    environment = dict(os.environ)
    environment["PWD"] = str(PROJECT_ROOT)
    environment["LOCAL_UID"] = str(getattr(os, "getuid", lambda: 0)())
    environment["LOCAL_GID"] = str(getattr(os, "getgid", lambda: 0)())
    return environment


def required_model_paths(backend_root: Path = BACKEND_ROOT) -> list[Path]:
    """Return complete runtime model and scaler matrix."""
    paths = [model_path(backend_root, model, mode, dataset)
             for model in MODEL_STEMS for mode in MODES for dataset in DATASETS]
    paths.extend(scaler_path(backend_root, model, mode, dataset)
                 for model in SCALER_MODELS for mode in MODES for dataset in DATASETS)
    return paths


def model_path(root: Path, model: str, mode: str, dataset: str) -> Path:
    """Build one model artifact path."""
    suffix = "" if dataset == "current" else "_url_corpus"
    filename = f"{MODEL_STEMS[model]}_{mode}{suffix}.pkl"
    return root / "models" / model / filename


def scaler_path(root: Path, model: str, mode: str, dataset: str) -> Path:
    """Build one scaler artifact path."""
    suffix = "" if dataset == "current" else "_url_corpus"
    filename = f"{model}_scaler_{mode}{suffix}.pkl"
    return root / "models" / model / filename


def required_dataset_paths(backend_root: Path = BACKEND_ROOT) -> list[Path]:
    """Return datasets required by runtime similarity search."""
    data_root = backend_root / "data"
    corpus_root = data_root / "url_corpus" / "quality_v4_repaired_features"
    return [
        data_root / "dataset_full.csv",
        data_root / "dataset_optimized.csv",
        corpus_root / "dataset_full_train.csv",
        corpus_root / "dataset_optimized_train.csv",
    ]


def validate_artifacts(backend_root: Path = BACKEND_ROOT) -> None:
    """Fail with exact paths when full bundle is incomplete."""
    required = [*required_model_paths(backend_root), *required_dataset_paths(backend_root)]
    missing = [path for path in required if not path.is_file()]
    if not missing:
        return
    names = "\n".join(f"- {path.relative_to(backend_root.parent)}" for path in missing)
    raise SetupError(f"Missing required project artifacts:\n{names}")


def validate_docker(runner: CommandRunner = run_command) -> None:
    """Verify Docker Compose availability."""
    result = runner(("docker", "compose", "version"), False, True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SetupError(f"Docker Compose unavailable: {detail}")


def certificates_ready(root: Path = CERTIFICATE_ROOT) -> bool:
    """Return whether canonical certificate pair exists."""
    return all((root / name).is_file() and (root / name).stat().st_size > 0
               for name in ("cert.pem", "key.pem"))


def prepare_certificates(runner: CommandRunner = run_command) -> None:
    """Generate canonical certificate pair when absent."""
    CERTIFICATE_ROOT.mkdir(parents=True, exist_ok=True)
    if certificates_ready():
        return
    runner(("docker", "compose", "run", "--build", "--rm", "certificates"), True, False)
    if not certificates_ready():
        raise SetupError("Certificate generation completed without cert.pem and key.pem")


def start_services(runner: CommandRunner = run_command) -> None:
    """Build and start Compose services with bounded failure logs."""
    result = runner(("docker", "compose", "up", "--build", "-d"), False, True)
    if result.returncode == 0:
        return
    logs = runner(("docker", "compose", "logs", "--tail", "80"), False, True)
    detail = result.stderr.strip() or result.stdout.strip()
    raise SetupError(f"Docker startup failed: {detail}\n{logs.stdout.strip()}")


def endpoint_ready(url: str) -> bool:
    """Check one local self-signed HTTPS endpoint."""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(url, context=context, timeout=3) as response:
            return 200 <= response.status < 400
    except OSError:
        return False


def wait_for_services(timeout: int = 120) -> None:
    """Wait until frontend and backend respond successfully."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(endpoint_ready(url) for url in SERVICE_URLS):
            return
        time.sleep(2)
    raise SetupError("Services did not become healthy within 120 seconds")


def wait_with_logs(
    waiter: Callable[[], None], runner: CommandRunner = run_command,
) -> None:
    """Attach recent Compose logs to health-check failures."""
    try:
        waiter()
    except SetupError as error:
        logs = runner(("docker", "compose", "logs", "--tail", "80"), False, True)
        raise SetupError(f"{error}\n{logs.stdout.strip()}") from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse setup execution mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check-only", action="store_true")
    group.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def execute(
    args: argparse.Namespace, runner: CommandRunner = run_command,
    waiter: Callable[[], None] = wait_for_services,
) -> None:
    """Execute selected setup workflow."""
    validate_docker(runner)
    validate_artifacts()
    if args.check_only:
        print("Docker Compose and project artifacts are ready.")
        return
    prepare_certificates(runner)
    if args.prepare_only:
        print(f"Certificates ready: {CERTIFICATE_ROOT}")
        return
    start_services(runner)
    wait_with_logs(waiter, runner)
    print("Frontend: https://localhost:4000")
    print("Backend:  https://localhost:5001")


def main() -> None:
    """Run setup and convert failures into concise CLI errors."""
    try:
        execute(parse_args())
    except (OSError, SetupError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
