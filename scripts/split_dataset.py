"""Divide o dataset convertido em treino, validação e teste sem fotógrafo compartilhado."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from epi.data.split import BALANCE_COLUMNS, assign_groups, group_totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/sh17_epi"))
    parser.add_argument("--split-csv", type=Path, default=Path("configs/sh17_epi_split.csv"))
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    return parser.parse_args()


def write_image_lists(index: pd.DataFrame, dataset_dir: Path) -> None:
    for split, rows in index.groupby("split"):
        lines = [f"./images/{stem}.jpg" for stem in sorted(rows["stem"])]
        (dataset_dir / f"{split}.txt").write_text("\n".join(lines) + "\n", newline="\n")


def summarize(index: pd.DataFrame, splits: list[str]) -> pd.DataFrame:
    count_columns = [column for column in BALANCE_COLUMNS if column != "images"]
    table = index.groupby("split")[count_columns].sum()
    table.insert(0, "images", index.groupby("split").size())
    table.insert(0, "photographers", index.groupby("split")["photographer_id"].nunique())
    return table.reindex(splits)


def main() -> None:
    args = parse_args()
    fractions = {"train": args.train, "val": args.val, "test": args.test}
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError(f"as frações precisam somar 1, somam {sum(fractions.values())}")

    index = pd.read_csv(args.dataset / "index.csv")
    assignment = assign_groups(group_totals(index, "photographer_id"), fractions)
    index["split"] = index["photographer_id"].map(assignment)

    if index.groupby("photographer_id")["split"].nunique().max() != 1:
        raise RuntimeError("um fotógrafo ficou em mais de um conjunto")

    write_image_lists(index, args.dataset)
    args.split_csv.parent.mkdir(parents=True, exist_ok=True)
    index[["stem", "photographer_id", "split"]].sort_values("stem").to_csv(
        args.split_csv, index=False, lineterminator="\n"
    )

    table = summarize(index, list(fractions))
    shares = (table / table.sum() * 100).round(1)
    print(table.to_string())
    print("\npercentual de cada coluna por conjunto:")
    print(shares.to_string())


if __name__ == "__main__":
    main()
