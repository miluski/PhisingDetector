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

from config.feature_names_config import FeatureConfig  # noqa: E402
from services.url_feature_service import URLFeatureService  # noqa: E402

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
    rows = []
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "url" not in reader.fieldnames:
            raise ValueError("Input CSV musi miec naglowek z kolumna 'url'")
        if "label" not in reader.fieldnames:
            raise ValueError("Input CSV musi miec kolumne 'label' (0/1)")
        for r in reader:
            url = (r.get("url") or "").strip()
            label_raw = (r.get("label") or "").strip()
            if not url or label_raw not in ("0", "1"):
                continue
            rows.append((url, int(label_raw)))
    return rows


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def extract_one(service: URLFeatureService, index: int, url: str, label: int,
                mode: str, network_mode: str):
    try:
        if network_mode == "live":
            features = service.extract(url, mode)
        else:
            features = service.extract_offline(url, mode)
        features["phishing"] = label
        return index, features
    except Exception as exc:
        logger.warning("Pominieto URL (blad ekstrakcji): %s -> %s", url, exc)
        return index, None


def build(input_path: Path, output_path: Path, mode: str, workers: int,
          network_mode: str, missing_policy: str, timeout: int,
          progress_interval: int, resume_temp: bool):
    socket.setdefaulttimeout(timeout)
    rows = load_urls(input_path)
    logger.info("Wczytano %d URL z %s", len(rows), input_path)
    if not rows:
        raise SystemExit("Brak prawidlowych URL w wejsciu.")

    feature_order = FeatureConfig.get_feature_names(mode)
    column_order = feature_order + ["phishing"]

    service = URLFeatureService(
        timeout=timeout,
        apply_network_defaults=missing_policy == "defaults"
    )

    done_indexes = set()
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    done_path = output_path.with_suffix(output_path.suffix + ".tmp.done")
    if resume_temp and tmp_path.exists() and done_path.exists():
        with done_path.open(encoding="utf-8") as f_done:
            done_indexes = {int(line) for line in f_done if line.strip().isdigit()}
        logger.info("Wznawiam temp: %d juz przetworzonych", len(done_indexes))

    pending_rows = [
        (index, url, label)
        for index, (url, label) in enumerate(rows)
        if index not in done_indexes
    ]

    done = len(done_indexes)
    written = count_csv_rows(tmp_path) if done_indexes else 0
    total = len(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (resume_temp and tmp_path.exists() and done_path.exists())
    csv_mode = "a" if not write_header else "w"
    with tmp_path.open(csv_mode, newline="", encoding="utf-8") as f, \
            done_path.open("a", encoding="utf-8") as f_done:
        writer = csv.DictWriter(f, fieldnames=column_order, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        pending = set()
        iterator = iter(pending_rows)
        max_pending = max(workers * 4, workers)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            def submit_next():
                try:
                    index, url, label = next(iterator)
                except StopIteration:
                    return False
                pending.add(pool.submit(extract_one, service, index, url, label, mode, network_mode))
                return True

            while len(pending) < max_pending and submit_next():
                pass

            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    index, row = fut.result()
                    done += 1
                    if row is not None:
                        writer.writerow({col: row.get(col, 0) for col in column_order})
                        f.flush()
                        written += 1
                    f_done.write(f"{index}\n")
                    f_done.flush()
                    if done % progress_interval == 0 or done == total:
                        logger.info("Postep: %d/%d | zapisano: %d", done, total, written)
                    submit_next()

    tmp_path.replace(output_path)
    done_path.unlink(missing_ok=True)
    logger.info("Zapisano %s (%d wierszy, %d kolumn)",
                output_path, written, len(column_order))


def parse_args():
    p = argparse.ArgumentParser(description="Build URL corpus dataset (project format)")
    p.add_argument("--input", required=True, type=Path,
                   help="CSV wejsciowy z kolumnami url,label")
    p.add_argument("--mode", choices=["optimized", "full"], default="full",
                   help="Tryb cech (full zaleca sie generowac raz, potem selekcja)")
    p.add_argument("--output", type=Path, default=None,
                   help="CSV wyjsciowy (domyslnie data/url_corpus/dataset_{mode}.csv)")
    p.add_argument("--workers", type=int, default=16,
                   help="Liczba watkow rownoleglej ekstrakcji")
    p.add_argument("--network-mode", choices=["defaults", "live"], default="defaults",
                   help="defaults = szybkie stabilne fallbacki, live = DNS/WHOIS/HTTP/SSL")
    p.add_argument("--missing-policy", choices=["defaults", "keep"], default="defaults",
                   help="defaults = zamien -1 na wartosci bazowe, keep = zachowaj -1")
    p.add_argument("--timeout", type=int, default=5,
                   help="Timeout sekund dla HTTP/DNS/SSL")
    p.add_argument("--progress-interval", type=int, default=1000,
                   help="Co ile rekordow logowac postep")
    p.add_argument("--resume-temp", action="store_true",
                   help="Wznow z dataset_full.csv.tmp i dataset_full.csv.tmp.done")
    return p.parse_args()


def main():
    args = parse_args()
    output = args.output or (BACKEND_DIR / "data" / "url_corpus" / f"dataset_{args.mode}.csv")
    build(args.input, output, args.mode, args.workers, args.network_mode, args.missing_policy,
          args.timeout, args.progress_interval, args.resume_temp)


if __name__ == "__main__":
    main()
