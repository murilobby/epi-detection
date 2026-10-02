"""Converte o SH17 bruto em um dataset YOLO com as classes de EPI do projeto."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import pandas as pd

from epi.data.convert import CLASSES, ConvertedSample, convert_sample
from epi.data.sh17 import Sample, load_samples, read_official_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/sh17"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/sh17_epi"))
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    return parser.parse_args()


def count_column(class_name: str) -> str:
    return f"n_{class_name.replace('-', '_')}"


def convert_all(
    samples: list[Sample], out_dir: Path, max_side: int, workers: int
) -> list[ConvertedSample]:
    results: dict[str, ConvertedSample] = {}
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(convert_sample, s, out_dir, max_side) for s in samples]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results[result.stem] = result
            if done % 500 == 0 or done == len(futures):
                print(f"{done}/{len(futures)}")
    return [results[sample.stem] for sample in samples]


def build_index(
    samples: list[Sample], results: list[ConvertedSample], official: dict[str, str]
) -> pd.DataFrame:
    rows = []
    for sample, result in zip(samples, results):
        counts = {count_column(name): 0 for name in CLASSES}
        for box in result.boxes:
            counts[count_column(CLASSES[box.class_id])] += 1
        rows.append(
            {
                "stem": sample.stem,
                "photographer_id": sample.photographer_id,
                "pexels_id": sample.pexels_id,
                "official_split": official[sample.stem],
                "orig_width": sample.width,
                "orig_height": sample.height,
                "width": result.width,
                "height": result.height,
                **counts,
                "dropped_boxes": result.dropped_boxes,
            }
        )
    return pd.DataFrame(rows)


def build_boxes(results: list[ConvertedSample]) -> pd.DataFrame:
    rows = [
        {
            "stem": result.stem,
            "class_name": CLASSES[box.class_id],
            "cx": box.cx,
            "cy": box.cy,
            "w": box.w,
            "h": box.h,
        }
        for result in results
        for box in result.boxes
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    for sub in ("images", "labels"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    # O paralelismo já é entre imagens; threads internas do OpenCV só competiriam pelos núcleos.
    cv2.setNumThreads(1)

    samples = load_samples(args.raw)
    official = read_official_split(args.raw)
    print(f"{len(samples)} imagens, {args.workers} threads, lado máximo {args.max_side}px")

    results = convert_all(samples, args.out, args.max_side, args.workers)
    index = build_index(samples, results, official)
    boxes = build_boxes(results)
    index.to_csv(args.out / "index.csv", index=False)
    boxes.to_csv(args.out / "boxes.csv", index=False)

    count_columns = [count_column(name) for name in CLASSES]
    print(boxes["class_name"].value_counts().reindex(CLASSES).to_string())
    print(f"caixas descartadas por ficarem vazias após o recorte: {index['dropped_boxes'].sum()}")
    print(f"imagens sem nenhuma das classes: {(index[count_columns].sum(axis=1) == 0).sum()}")


if __name__ == "__main__":
    main()
