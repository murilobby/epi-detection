"""Procura imagens quase duplicadas e conta quantos pares atravessam os conjuntos."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import pandas as pd

from epi.data.dedup import close_pairs, hash_files

THRESHOLDS = (0, 2, 4, 6, 8, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/sh17_epi"))
    parser.add_argument("--split-csv", type=Path, default=Path("configs/sh17_epi_split.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/near_duplicates.csv"))
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    return parser.parse_args()


def label_pairs(pairs: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    a = index.iloc[pairs["i"]].reset_index(drop=True)
    b = index.iloc[pairs["j"]].reset_index(drop=True)
    return pd.DataFrame(
        {
            "stem_a": a["stem"],
            "stem_b": b["stem"],
            "distance": pairs["distance"].to_numpy(),
            "same_photographer": (a["photographer_id"] == b["photographer_id"]).to_numpy(),
            "split_a": a["split"],
            "split_b": b["split"],
            "official_a": a["official_split"],
            "official_b": b["official_split"],
        }
    )


def count_by_threshold(pairs: pd.DataFrame) -> pd.DataFrame:
    crosses = pairs["split_a"] != pairs["split_b"]
    crosses_official = pairs["official_a"] != pairs["official_b"]
    rows = []
    for threshold in THRESHOLDS:
        within = pairs["distance"] <= threshold
        rows.append(
            {
                "max_distance": threshold,
                "pairs": int(within.sum()),
                "cross_split": int((within & crosses).sum()),
                "cross_official_split": int((within & crosses_official).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(1)

    index = pd.read_csv(args.dataset / "index.csv").merge(
        pd.read_csv(args.split_csv)[["stem", "split"]], on="stem", validate="one_to_one"
    )
    paths = [args.dataset / "images" / f"{stem}.jpg" for stem in index["stem"]]
    hashes = hash_files(paths, args.workers)

    pairs = label_pairs(close_pairs(hashes, max(THRESHOLDS)), index)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pairs.sort_values(["distance", "stem_a"]).to_csv(args.out, index=False)

    print(f"{len(index)} imagens, {len(pairs)} pares com distância <= {max(THRESHOLDS)} de 64 bits")
    print(count_by_threshold(pairs).to_string(index=False))


if __name__ == "__main__":
    main()
