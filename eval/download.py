"""Dataset downloader: fetch the file named in a frozen config, verify its
sha256. If the config has no checksum yet, print the computed one to paste
into the config (first-time pinning).

Usage: uv run python -m eval.download eval/configs/longmemeval_s.json
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from eval.run import ROOT, load_config, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="eval/configs/<benchmark>.json")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = config["dataset"]
    target = ROOT / dataset["file"]
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and dataset.get("sha256") == sha256_file(target):
        print(f"OK (déjà présent, checksum conforme): {target}")
        return 0

    print(f"téléchargement: {dataset['url']}\n  -> {target}")
    with urllib.request.urlopen(dataset["url"], timeout=60) as response, target.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)

    actual = sha256_file(target)
    expected = dataset.get("sha256")
    if not expected:
        print(f"pas de checksum dans la config — épingle cette valeur :\n  \"sha256\": \"{actual}\"")
        return 0
    if actual != expected:
        target.unlink()
        print(f"CHECKSUM MISMATCH: attendu {expected}, obtenu {actual} (fichier supprimé)")
        return 1
    print(f"OK, checksum conforme: {actual}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
