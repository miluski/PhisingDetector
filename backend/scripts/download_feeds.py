"""Download phishing and legit URL feeds into raw url,label CSV."""

import argparse
import csv
import io
import logging
import urllib.request
import zipfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_feeds")

PHISHING_DB_LINKS = (
    "https://raw.githubusercontent.com/Phishing-Database/"
    "Phishing.Database/master/phishing-links-ACTIVE.txt"
)
TRANCO_TOP = "https://tranco-list.eu/top-1m.csv.zip"
USER_AGENT = "Mozilla/5.0 (research; phishing-detector thesis)"

csv.field_size_limit(10 * 1024 * 1024)


def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_phishing_db(feed_url: str, limit: int) -> list:
    """Download active phishing URLs from the configured feed."""
    logger.info("Pobieram Phishing.Database (MIT): %s", feed_url)
    raw = _fetch(feed_url).decode("utf-8", errors="ignore")
    urls = [ln.strip() for ln in raw.splitlines() if ln.strip().startswith("http")]
    logger.info("Phishing URL (active): %d", len(urls))
    return urls[:limit]


def fetch_legit(limit: int) -> list:
    """Download HTTPS roots from the Tranco top list."""
    logger.info("Pobieram Tranco top list: %s", TRANCO_TOP)
    raw = _fetch(TRANCO_TOP)
    urls = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8")
            for row in csv.reader(text):
                if len(row) >= 2 and row[1]:
                    urls.append(f"https://{row[1].strip()}/")
                if len(urls) >= limit:
                    break
    logger.info("Legit URL pobranych: %d", len(urls))
    return urls


def build(output: Path, limit: int, phish_url: str):
    """Download both feeds and write a balanced raw corpus."""
    phishing = fetch_phishing_db(phish_url, limit)
    legit = fetch_legit(limit)

    n = min(len(phishing), len(legit))
    if n == 0:
        raise SystemExit("Brak danych w ktoryms feedzie — przerwano.")
    phishing, legit = phishing[:n], legit[:n]
    logger.info("Balans: %d phishing + %d legit = %d wierszy", n, n, 2 * n)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "label"])
        for u in phishing:
            w.writerow([u, 1])
        for u in legit:
            w.writerow([u, 0])
    logger.info("Zapisano %s (%d wierszy)", output, 2 * n)


def parse_args():
    """Parse feed download options."""
    p = argparse.ArgumentParser(description="Download phishing+legit feeds -> raw_urls.csv")
    p.add_argument("--limit", type=int, default=60000,
                   help="Maks URL na klase (phishing i legit osobno)")
    p.add_argument("--output", type=Path,
                   default=BACKEND_DIR / "data" / "url_corpus" / "raw_urls.csv")
    p.add_argument("--phish-url", default=None,
                   help="Override URL feedu Phishing.Database")
    return p.parse_args()


def main():
    """Download feeds using parsed command-line options."""
    args = parse_args()
    build(args.output, args.limit, args.phish_url or PHISHING_DB_LINKS)


if __name__ == "__main__":
    main()
