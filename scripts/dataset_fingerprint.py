"""Calcula uma impressão digital SHA-256 de cada parte do dataset processado.

Serve para comparar o dataset gerado em máquinas diferentes. Arquivos de texto têm a quebra de
linha normalizada para LF, para que a comparação não dependa do sistema operacional.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/sh17_epi"))
    return parser.parse_args()


def file_digest(path: Path, is_text: bool) -> str:
    data = path.read_bytes()
    if is_text:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def folder_digest(folder: Path, pattern: str, is_text: bool) -> tuple[int, str]:
    """Combina, em ordem de nome, o nome e o hash de cada arquivo da pasta."""
    combined = hashlib.sha256()
    files = sorted(folder.glob(pattern))
    for path in files:
        combined.update(f"{path.name}:{file_digest(path, is_text)}\n".encode())
    return len(files), combined.hexdigest()


def main() -> None:
    args = parse_args()
    for name in ("index.csv", "boxes.csv", "train.txt", "val.txt", "test.txt"):
        print(f"{name:12s} {file_digest(args.dataset / name, is_text=True)}")
    for folder, pattern, is_text in (("labels", "*.txt", True), ("images", "*.jpg", False)):
        count, digest = folder_digest(args.dataset / folder, pattern, is_text)
        print(f"{folder:12s} {digest}  ({count} arquivos)")


if __name__ == "__main__":
    main()
