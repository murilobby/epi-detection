"""Calibra o tamanho mínimo de cabeça para o sistema afirmar que alguém está sem EPI.

Usa a validação: para cada capacete anotado sobre uma cabeça anotada, verifica se o detector o
encontrou e agrupa pelo tamanho da cabeça, que é o que o sistema enxerga quando o capacete não é
detectado. O mínimo é o início da menor faixa a partir da qual o recall de capacete fica acima do
alvo em todas as faixas maiores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epi.eval.errors import XYXY
from epi.track.ppe import coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=Path("reports/eval/yolov8s_640_val/matches.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/track/head_size_calibration.csv"))
    parser.add_argument("--target", type=float, default=0.5, help="recall mínimo de capacete")
    # Faixas de 16 px: com faixas de 8 px, a de 24 a 32 px tinha só 6 capacetes e decidia o limiar sozinha.
    parser.add_argument("--edges", type=float, nargs="+", default=[0, 16, 32, 48, 64, 96, np.inf])
    return parser.parse_args()


def helmets_on_heads(matches: pd.DataFrame) -> pd.DataFrame:
    """Cada capacete anotado com pelo menos metade da área dentro de uma cabeça anotada."""
    rows = []
    gt = matches[matches.source == "gt"]
    for _, image in gt.groupby("stem"):
        helmets = image[image.class_name == "helmet"]
        heads = image[image.class_name == "head"]
        if helmets.empty or heads.empty:
            continue
        cov = coverage(helmets[XYXY].to_numpy(), heads[XYXY].to_numpy())
        best = cov.argmax(axis=1)
        for k, (outcome, inside) in enumerate(zip(helmets.outcome, cov.max(axis=1))):
            if inside >= 0.5:
                rows.append({"found": outcome == "TP", "head_side_px": heads.side_px.iloc[best[k]]})
    return pd.DataFrame(rows)


def recall_by_head_size(pairs: pd.DataFrame, edges: list[float]) -> pd.DataFrame:
    bins = pd.cut(pairs.head_side_px, edges, right=False)
    grouped = pairs.found.groupby(bins, observed=False)
    table = pd.DataFrame({"helmets": grouped.size(), "found": grouped.sum()})
    table["recall"] = table.found / table.helmets.replace(0, np.nan)
    return table


def min_head_side(table: pd.DataFrame, target: float) -> float:
    ok = (table.recall >= target).to_numpy()
    # Percorre de trás para frente: a faixa vale se ela e todas as maiores atingem o alvo.
    start = len(ok)
    while start > 0 and ok[start - 1]:
        start -= 1
    if start == len(ok):
        raise ValueError("nenhuma faixa atinge o recall alvo")
    return float(table.index[start].left)


def main() -> None:
    args = parse_args()
    pairs = helmets_on_heads(pd.read_csv(args.matches))
    table = recall_by_head_size(pairs, args.edges)
    threshold = min_head_side(table, args.target)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.out, lineterminator="\n")
    print(f"{len(pairs)} capacetes sobre cabeças anotadas")
    print(table.round(3).to_string())
    print(f"tamanho mínimo de cabeça para recall de capacete >= {args.target}: {threshold:g} px")


if __name__ == "__main__":
    main()
