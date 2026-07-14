"""Fetch and verify the authors-released FOR-instance-clean TreeLearn checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path


SOURCE_URL = (
    "https://data.goettingen-research-online.de/api/access/datafile/"
    ":persistentId?persistentId=doi:10.25625/VPMPID/8CIIW0"
)
EXPECTED_MD5 = "106a80de2991c5f23484a3f9d03e3b16"


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(output: Path) -> str:
    output = output.expanduser().resolve()
    if output.exists():
        if not output.is_file() or md5(output) != EXPECTED_MD5:
            raise ValueError(f"Existing clean checkpoint has wrong MD5: {output}")
        return "reused_verified"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.part.{os.getpid()}")
    try:
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
            with temporary.open("xb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
        if md5(temporary) != EXPECTED_MD5:
            raise ValueError("Downloaded clean checkpoint MD5 mismatch")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return "downloaded_verified"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    status = fetch(args.output)
    print(f"checkpoint_status={status}")
    print(f"checkpoint={args.output.expanduser().resolve()}")
    print(f"checkpoint_md5={EXPECTED_MD5}")
    print("checkpoint_persistent_id=doi:10.25625/VPMPID/8CIIW0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
