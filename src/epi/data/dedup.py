"""Detecção de imagens quase duplicadas com difference hash (dHash)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HASH_SIZE = 8


def dhash(gray: np.ndarray, hash_size: int = HASH_SIZE) -> np.uint64:
    """Reduz a imagem a (hash_size + 1) x hash_size e codifica se cada pixel é mais claro que o vizinho."""
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.packbits(bits).view(">u8")[0].astype(np.uint64)


def hash_file(path: Path) -> np.uint64:
    # REDUCED_GRAYSCALE_4 decodifica o JPEG já em 1/4 da resolução, o que basta para um hash 9x8.
    gray = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if gray is None:
        raise ValueError(f"não foi possível ler {path}")
    return dhash(gray)


def hash_files(paths: list[Path], workers: int) -> np.ndarray:
    with ThreadPoolExecutor(workers) as pool:
        return np.array(list(pool.map(hash_file, paths)), dtype=np.uint64)


def close_pairs(hashes: np.ndarray, max_distance: int, chunk: int = 1024) -> pd.DataFrame:
    """Todos os pares i < j com distância de Hamming até max_distance, comparando em blocos de linhas."""
    frames = []
    for start in range(0, len(hashes), chunk):
        block = hashes[start : start + chunk]
        distance = np.bitwise_count(block[:, None] ^ hashes[None, :])
        rows, cols = np.nonzero(distance <= max_distance)
        keep = rows + start < cols
        rows, cols = rows[keep], cols[keep]
        frames.append(
            pd.DataFrame({"i": rows + start, "j": cols, "distance": distance[rows, cols]})
        )
    return pd.concat(frames, ignore_index=True)
