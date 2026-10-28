"""Unpack the gzipped data mirror and verify every file against Dryad's checksums.

Files already present in data/ (e.g. downloaded by hand from Dryad) are not overwritten,
only verified. Exits non-zero if any file is missing or has the wrong checksum.

Usage: python data/prepare_data.py
"""

import gzip
import hashlib
import shutil
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
MIRROR_DIR = DATA_DIR / "mirror"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    bad = 0
    for line in (DATA_DIR / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split()
        target = DATA_DIR / name
        if not target.exists():
            src = MIRROR_DIR / f"{name}.gz"
            if not src.exists():
                print(f"MISSING  {name} (not in data/ or data/mirror/)")
                bad += 1
                continue
            with gzip.open(src, "rb") as fin, open(target, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        ok = sha256(target) == expected
        bad += not ok
        print(f"{'OK      ' if ok else 'BAD SUM '} {name}")
    if bad:
        sys.exit(f"{bad} file(s) missing or failed the checksum")
    print("All files match the Dryad checksums.")


if __name__ == "__main__":
    main()
