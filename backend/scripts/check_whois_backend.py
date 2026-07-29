"""Check RDAP/WHOIS availability for domains before opting into model use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from extractors.network.whois_extractor import WHOISExtractor


def parse_args() -> argparse.Namespace:
    """Parse WHOIS/RDAP probe options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["example.com", "google.com", "github.com"],
        help="Domains to query (default: three stable public domains).",
    )
    parser.add_argument("--timeout", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """Probe configured domains and print completeness statistics."""
    args = parse_args()
    extractor = WHOISExtractor(timeout=args.timeout)
    results = []
    for domain in args.domains:
        values = extractor.extract_all(domain)
        results.append(
            {
                "domain": domain,
                **values,
                "complete": all(value != -1 for value in values.values()),
            }
        )
    complete = sum(int(row["complete"]) for row in results)
    print(json.dumps({"complete": complete, "total": len(results), "results": results}, indent=2))
    if complete == 0:
        raise SystemExit("WHOIS/RDAP returned no usable dates for any domain")


if __name__ == "__main__":
    main()
