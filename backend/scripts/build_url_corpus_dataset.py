"""Build URL corpus dataset from raw url,label CSV."""

import argparse
import csv
import logging
import socket
import sys
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from config.feature_names_config import FeatureConfig
from config.feature_modes import SUPPORTED_MODES
from config.feature_policy import AUDIT_INDICATORS, MISSING_POLICY, NETWORK_MODE, add_missing_indicators
from services.url_feature_service import URLFeatureService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_url_corpus")

warnings.filterwarnings("ignore", category=UserWarning, module="bs4")
warnings.filterwarnings("ignore", message="Some characters could not be decoded.*")
logging.getLogger("URLFeatureService").setLevel(logging.WARNING)
for noisy_logger in (
    "DNSResolver",
    "ResponseTimer",
    "extractors.network.dns_resolver",
    "extractors.network.response_timer",
    "extractors.network.ssl_validator",
    "extractors.network.whois_extractor",
    "urllib3",
    "whois",
):
    logging.getLogger(noisy_logger).setLevel(logging.CRITICAL)


def load_urls(input_path: Path):
    """Load valid URL/label rows from the input CSV."""
    rows = []
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "url" not in reader.fieldnames:
            raise ValueError("Input CSV musi miec naglowek z kolumna 'url'")
        if "label" not in reader.fieldnames:
            raise ValueError("Input CSV musi miec kolumne 'label' (0/1)")
        for fallback_index, r in enumerate(reader):
            url = (r.get("url") or "").strip()
            label_raw = (r.get("label") or "").strip()
            if not url or label_raw not in ("0", "1"):
                continue
            source_index_raw = (r.get("source_index") or "").strip()
            source_index = int(source_index_raw) if source_index_raw.isdigit() else fallback_index
            rows.append((source_index, url, int(label_raw)))
    return rows


def count_csv_rows(path: Path) -> int:
    """Count data rows in an existing CSV, returning zero when absent."""
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def extract_one(service: URLFeatureService, index: int, url: str, label: int,
                mode: str, network_mode: str, source_index: int, include_metadata: bool,
                failure_policy: str, feature_order: list[str]):
    """Extract one row and apply the configured failure policy."""
    try:
        return index, _extract_features(service, url, label, mode, network_mode,
                                        source_index, include_metadata)
    except Exception as exc:
        if failure_policy == "keep-empty":
            logger.warning("Zachowano URL z pustymi cechami (blad ekstrakcji): %s -> %s", url, exc)
            return index, _empty_features(feature_order, url, label, source_index, include_metadata)
        logger.warning("Pominieto URL (blad ekstrakcji): %s -> %s", url, exc)
        return index, None


def _extract_features(service, url, label, mode, network_mode, source_index, include_metadata):
    """Extract and annotate one successful feature dictionary."""
    features = service.extract(url, mode) if network_mode == "live" else service.extract_offline(url, mode)
    features = add_missing_indicators(features)
    if include_metadata:
        features.update({"source_index": source_index, "url": url})
    features["phishing"] = label
    return features


def _empty_features(feature_order, url, label, source_index, include_metadata):
    """Create a policy-compliant row when extraction fails."""
    features = add_missing_indicators({feature: -1 for feature in feature_order})
    if include_metadata:
        features.update({"source_index": source_index, "url": url})
    features["phishing"] = label
    return features


def _validate_options(network_mode, missing_policy):
    """Validate network and missing-value policies before extraction."""
    if network_mode not in {"live", "offline"}:
        raise ValueError(f"Unsupported network mode: {network_mode}")
    if missing_policy != "keep":
        raise ValueError("Użyj --missing-policy keep dla wspólnego schematu cech")


def _resume_state(output_path, resume_temp):
    """Return temporary paths and completed row indexes for resume mode."""
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    done_path = output_path.with_suffix(output_path.suffix + ".tmp.done")
    if not resume_temp or not tmp_path.exists() or not done_path.exists():
        return tmp_path, done_path, set()
    with done_path.open(encoding="utf-8") as handle:
        done = {int(line) for line in handle if line.strip().isdigit()}
    logger.info("Wznawiam temp: %d juz przetworzonych", len(done))
    return tmp_path, done_path, done


def _submit_next(iterator, pending, pool, service, mode, network_mode, include_metadata,
                 failure_policy, feature_order):
    """Submit one extraction job and report whether a row was available."""
    try:
        index, source_index, url, label = next(iterator)
    except StopIteration:
        return False
    pending.add(pool.submit(extract_one, service, index, url, label, mode, network_mode,
                            source_index, include_metadata, failure_policy, feature_order))
    return True


def _consume_finished(finished, writer, stream, done_file, column_order, progress, total, done, written):
    """Persist completed futures and return updated counters."""
    for future in finished:
        index, row = future.result()
        done += 1
        if row is not None:
            writer.writerow({column: row.get(column, -1) for column in column_order})
            stream.flush()
            written += 1
        done_file.write(f"{index}\n")
        done_file.flush()
        if done % progress == 0 or done == total:
            logger.info("Postep: %d/%d | zapisano: %d", done, total, written)
    return done, written


def _run_workers(rows, total, service, mode, network_mode, workers, progress_interval,
                 include_metadata, failure_policy, feature_order, column_order,
                 writer, stream, done_file, initial_done, initial_written):
    """Run bounded extraction workers and persist results incrementally."""
    pending = set()
    iterator = iter(rows)
    pool = ThreadPoolExecutor(max_workers=workers)
    limit = max(workers * 4, workers)
    done, written = initial_done, initial_written
    try:
        done, written = _drain_workers(
            iterator, pending, pool, limit, service, mode, network_mode,
            include_metadata, failure_policy, feature_order, writer, stream, done_file,
            column_order, progress_interval, total, done, written,
        )
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return written


def _drain_workers(iterator, pending, pool, limit, service, mode, network_mode,
                   include_metadata, failure_policy, feature_order, writer, stream,
                   done_file, column_order, progress, total, done, written):
    """Submit, consume and refill futures until all rows finish."""
    _fill_pending(iterator, pending, pool, limit, service, mode, network_mode,
                  include_metadata, failure_policy, feature_order)
    while pending:
        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
        done, written = _consume_finished(finished, writer, stream, done_file, column_order,
                                          progress, total, done, written)
        _fill_pending(iterator, pending, pool, limit, service, mode, network_mode,
                      include_metadata, failure_policy, feature_order)
    return done, written


def _fill_pending(iterator, pending, pool, limit, service, mode, network_mode,
                  include_metadata, failure_policy, feature_order):
    """Fill the bounded future queue until the configured limit."""
    while len(pending) < limit and _submit_next(iterator, pending, pool, service, mode, network_mode,
                                                include_metadata, failure_policy, feature_order):
        pass


def build(input_path: Path, output_path: Path, mode: str, workers: int,
          network_mode: str, missing_policy: str, timeout: int,
          progress_interval: int, resume_temp: bool, include_metadata: bool,
          failure_policy: str):
    """Build a feature CSV with resumable bounded parallel extraction."""
    socket.setdefaulttimeout(timeout)
    rows = load_urls(input_path)
    logger.info("Wczytano %d URL z %s", len(rows), input_path)
    if not rows:
        raise SystemExit("Brak prawidlowych URL w wejsciu.")
    _validate_options(network_mode, missing_policy)
    config = _build_context(output_path, mode, timeout, resume_temp, include_metadata)
    pending_rows = [(i, *row) for i, row in enumerate(rows) if i not in config["completed"]]
    written = _write_dataset(config, pending_rows, len(rows), workers, network_mode,
                             progress_interval, failure_policy, len(config["completed"]))
    tmp_path, done_path, column_order = config["tmp_path"], config["done_path"], config["column_order"]
    tmp_path.replace(output_path)
    done_path.unlink(missing_ok=True)
    logger.info("Zapisano %s (%d wierszy, %d kolumn)", output_path, written, len(column_order))


def _build_context(output_path, mode, timeout, resume_temp, include_metadata):
    """Build feature service, schema and resumable output paths."""
    feature_order = FeatureConfig.get_feature_names(mode)
    metadata = ["source_index", "url"] if include_metadata else []
    tmp_path, done_path, completed = _resume_state(output_path, resume_temp)
    return {"service": URLFeatureService(timeout=timeout, apply_network_defaults=False), "mode": mode,
            "include_metadata": include_metadata,
            "feature_order": feature_order,
            "column_order": metadata + feature_order + list(AUDIT_INDICATORS) + ["phishing"],
            "tmp_path": tmp_path, "done_path": done_path, "completed": completed}


def _write_dataset(config, pending_rows, total, workers, network_mode, progress, failure_policy, initial_done):
    """Open resumable files and run bounded feature extraction."""
    config["tmp_path"].parent.mkdir(parents=True, exist_ok=True)
    append = bool(initial_done and config["tmp_path"].exists() and config["done_path"].exists())
    mode_flag = "a" if append else "w"
    with config["tmp_path"].open(mode_flag, newline="", encoding="utf-8") as data_file, \
            config["done_path"].open("a", encoding="utf-8") as done_file:
        writer = csv.DictWriter(data_file, fieldnames=config["column_order"], extrasaction="ignore")
        if not append:
            writer.writeheader()
        written = count_csv_rows(config["tmp_path"]) if initial_done else 0
        return _run_workers(pending_rows, total, config["service"], config["mode"], network_mode, workers, progress,
                            config["include_metadata"], failure_policy, config["feature_order"], config["column_order"],
                            writer, data_file, done_file, initial_done, written)


def parse_args():
    """Parse URL corpus extraction command-line arguments."""
    return _argument_parser().parse_args()


def _argument_parser():
    """Build the URL corpus extraction argument parser."""
    p = argparse.ArgumentParser(description="Build URL corpus dataset (project format)")
    _add_input_arguments(p)
    _add_runtime_arguments(p)
    _add_failure_arguments(p)
    return p


def _add_input_arguments(parser):
    """Add input, mode and output arguments."""
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default="full")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-metadata", action="store_true")


def _add_runtime_arguments(parser):
    """Add worker, network and progress arguments."""
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--network-mode", choices=["live", "offline"], default=NETWORK_MODE)
    parser.add_argument("--missing-policy", choices=["keep"], default=MISSING_POLICY)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--resume-temp", action="store_true")


def _add_failure_arguments(parser):
    """Add extraction error policy argument."""
    parser.add_argument("--failure-policy", choices=["skip", "keep-empty"], default="skip")


def main():
    """Parse arguments and execute one dataset build."""
    args = parse_args()
    build(args.input, args.output, args.mode, args.workers, args.network_mode, args.missing_policy,
          args.timeout, args.progress_interval, args.resume_temp, args.include_metadata,
          args.failure_policy)


if __name__ == "__main__":
    main()
