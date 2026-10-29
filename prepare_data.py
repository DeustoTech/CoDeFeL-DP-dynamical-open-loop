"""Download or verify Fashion-MNIST only when explicitly invoked."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "Control_Theoretic_DP_Numerics"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="Verify local files without downloading anything")
    args = parser.parse_args()
    sys.path.insert(0, str(CORE / "code"))
    import fashion_mnist_mlp_schedule_compare as model

    directory = CORE / "data/fashion_mnist"
    if not args.check_only:
        for url in model.URLS.values():
            model.download_file(url, directory / Path(url).name)
    hashes = {}
    for name, expected in model.EXPECTED_SHA256.items():
        path = directory / name
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Supply the four original archives, "
                             "or run python prepare_data.py to download them.")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(f"Checksum mismatch: {path}")
        hashes[str(path.relative_to(ROOT))] = digest
        print(f"Verified {name}")
    # The original solver audit expects this provenance file. It is generated
    # from the actual recipient-side files, rather than shipped with results.
    manifest = ROOT / "experiments/solver_comparison_2026_10_08/data_sha256.json"
    manifest.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
