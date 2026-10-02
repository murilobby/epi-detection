"""Gera tabelas e gráficos de análise do dataset convertido e dividido."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from epi.data.analysis import (
    MEDIUM_MAX,
    SMALL_MAX,
    SPLITS,
    box_relations,
    class_summary,
    concentration_curve,
    image_summary,
    load_tables,
    photographer_concentration,
    pixel_side,
    size_buckets,
)
from epi.data.convert import CLASSES
from epi.viz import (
    AXIS,
    INK_SECONDARY,
    MUTED,
    SERIES,
    new_figure,
    set_title,
    style_legend,
    thousands,
)

SPLIT_LABELS = {"train": "treino", "val": "validação", "test": "teste"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/sh17_epi"))
    parser.add_argument("--split-csv", type=Path, default=Path("configs/sh17_epi_split.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/dataset"))
    return parser.parse_args()


def plot_class_instances(classes: pd.DataFrame, path: Path) -> None:
    figure, ax = new_figure(height=3.0)
    counts = classes["instances"]
    positions = np.arange(len(counts))[::-1]
    ax.barh(positions, counts, height=0.5, color=SERIES[0])
    for position, value in zip(positions, counts):
        ax.text(value + counts.max() * 0.01, position, thousands(value),
                va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(positions, counts.index)
    ax.set_xlim(0, counts.max() * 1.12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: thousands(value)))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("caixas anotadas")
    set_title(ax, "Instâncias por classe", "SH17 convertido, todos os conjuntos")
    figure.savefig(path)


def plot_box_sizes(boxes: pd.DataFrame, imgsz: int, path: Path) -> None:
    figure, ax = new_figure()
    side = pixel_side(boxes, imgsz)
    for color, name in zip(SERIES, CLASSES):
        values = np.sort(side[boxes["class_name"] == name].to_numpy())
        ax.step(values, np.arange(1, len(values) + 1) / len(values) * 100, where="post",
                color=color, linewidth=2, label=f"{name} (n={thousands(len(values))})")
    for limit in (SMALL_MAX, MEDIUM_MAX):
        ax.axvline(limit, color=AXIS, linewidth=1)
    for x, label in ((8, "pequeno"), (55, "médio"), (300, "grande")):
        ax.text(x, 101, label, ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(2, 1300)
    ax.set_xticks([2, 5, 10, 20, 50, 100, 200, 500, 1000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: thousands(value)))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda value, _: ""))
    ax.set_ylim(0, 100)
    ax.set_xlabel(f"lado equivalente da caixa (px), imagem com lado maior de {imgsz} px")
    ax.set_ylabel("% das caixas com lado até x")
    set_title(ax, f"Tamanho das caixas em {imgsz} px",
              f"distribuição acumulada; limites do COCO em {SMALL_MAX} e {MEDIUM_MAX} px")
    style_legend(ax, loc="lower right")
    figure.savefig(path)


def plot_concentration(index: pd.DataFrame, path: Path) -> None:
    figure, ax = new_figure()
    ax.plot([0, 100], [0, 100], color=AXIS, linewidth=1)
    ax.text(62, 55, "todos com o mesmo número de fotos", rotation=33, fontsize=8, color=MUTED)
    for color, split in zip(SERIES, SPLITS):
        x, y = concentration_curve(index, split)
        ax.plot(x, y, color=color, linewidth=2, label=SPLIT_LABELS[split])
    ax.text(24, 66, "validação e teste quase coincidem", fontsize=9, color=INK_SECONDARY)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("% dos fotógrafos, do que tem mais fotos para o que tem menos")
    ax.set_ylabel("% acumulado das imagens")
    set_title(ax, "Concentração de imagens por fotógrafo",
              "curva alta à esquerda: poucas fontes dominam o conjunto")
    style_legend(ax, loc="lower right")
    figure.savefig(path)


def save_table(table: pd.DataFrame, path: Path, title: str) -> None:
    table.to_csv(path, lineterminator="\n")
    print(f"\n{title}\n{table.to_string()}")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index, boxes = load_tables(args.dataset, args.split_csv)

    classes = class_summary(boxes)
    save_table(classes, args.out / "class_summary.csv", "classes")
    save_table(image_summary(index), args.out / "image_summary.csv", "imagens")
    sizes = pd.concat([size_buckets(boxes, imgsz) for imgsz in (640, 1280)])
    save_table(sizes, args.out / "box_sizes.csv", "tamanho das caixas (% por faixa do COCO)")
    save_table(box_relations(boxes), args.out / "box_relations.csv", "relações entre caixas")
    save_table(photographer_concentration(index), args.out / "photographers.csv", "fotógrafos")

    plot_class_instances(classes, args.out / "class_instances.png")
    plot_box_sizes(boxes, 640, args.out / "box_sizes_640.png")
    plot_concentration(index, args.out / "photographer_concentration.png")
    print(f"\ntabelas e gráficos em {args.out}")


if __name__ == "__main__":
    main()
