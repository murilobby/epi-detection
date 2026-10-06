"""Analisa os erros do detector no teste: recall por tamanho, causas dos erros e exemplos visuais."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from epi.eval.errors import (
    SIZE_LABELS,
    XYXY,
    false_negative_causes,
    false_positive_causes,
    recall_by_size,
)
from epi.viz import INK_SECONDARY, new_figure, set_title, style_legend

# Rampa sequencial em azul para as faixas de tamanho, que são ordenadas.
SIZE_COLORS = ("#86b6ef", "#3987e5", "#184f95")
GT_BGR, PRED_BGR = (122, 175, 27), (52, 104, 235)
TILE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=Path("reports/eval/yolov8s_640"))
    parser.add_argument("--images", type=Path, default=Path("data/processed/sh17_epi/images"))
    parser.add_argument("--examples", type=int, default=6)
    return parser.parse_args()


def plot_recall_by_size(table: pd.DataFrame, path: Path) -> None:
    classes = list(dict.fromkeys(table.index.get_level_values(0)))
    figure, ax = new_figure(height=3.8)
    width = 0.26
    for k, (size, color) in enumerate(zip(SIZE_LABELS, SIZE_COLORS)):
        rows = table.xs(size, level=1).reindex(classes)
        positions = np.arange(len(classes)) + (k - 1) * width
        ax.bar(positions, rows.recall.fillna(0), width=width * 0.9, color=color, label=size)
        for x, (recall, n) in zip(positions, zip(rows.recall, rows.instances)):
            if n:
                ax.text(x, recall + 0.02, f"n={n}", ha="center", fontsize=7, color=INK_SECONDARY)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("recall no teste")
    ax.grid(axis="x", visible=False)
    set_title(ax, "Recall por tamanho do objeto", "faixas do COCO com a imagem em 640 px: pequeno < 32 px <= médio < 96 px")
    style_legend(ax, loc="upper left", ncols=3)
    figure.savefig(path)


def crop_tile(image: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, tuple[int, int, float]]:
    """Recorte quadrado ao redor da caixa, com contexto, redimensionado para TILE x TILE."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box * [w, h, w, h]
    side = int(min(max(3 * (x1 - x0), 3 * (y1 - y0), 96), max(w, h)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(np.clip(cx - side / 2, 0, max(w - side, 0)))
    top = int(np.clip(cy - side / 2, 0, max(h - side, 0)))
    crop = image[top : top + side, left : left + side]
    scale = TILE / max(crop.shape[:2])
    tile = cv2.resize(crop, None, fx=scale, fy=scale)
    canvas = np.full((TILE, TILE, 3), 255, np.uint8)
    canvas[: tile.shape[0], : tile.shape[1]] = tile
    return canvas, (left, top, scale)


def draw_boxes(tile: np.ndarray, rows: pd.DataFrame, color: tuple[int, int, int], size: tuple[int, int],
               offset: tuple[int, int, float], thickness: int) -> None:
    w, h = size
    left, top, scale = offset
    for x0, y0, x1, y1 in rows[XYXY].to_numpy() * [w, h, w, h]:
        p0 = (int((x0 - left) * scale), int((y0 - top) * scale))
        p1 = (int((x1 - left) * scale), int((y1 - top) * scale))
        cv2.rectangle(tile, p0, p1, color, thickness)


def example_grid(examples: pd.DataFrame, matches: pd.DataFrame, images: Path, path: Path, per_row: int = 0) -> None:
    tiles = []
    for row in examples.itertuples():
        image = cv2.imread(str(images / f"{row.stem}.jpg"))
        tile, offset = crop_tile(image, np.array([row.x0, row.y0, row.x1, row.y1]))
        same_image = matches[matches.stem == row.stem]
        size = (image.shape[1], image.shape[0])
        draw_boxes(tile, same_image[same_image.source == "gt"], GT_BGR, size, offset, 1)
        draw_boxes(tile, same_image[same_image.source == "pred"], PRED_BGR, size, offset, 1)
        highlight = GT_BGR if row.source == "gt" else PRED_BGR
        draw_boxes(tile, examples.loc[[row.Index]], highlight, size, offset, 3)
        label = f"{row.outcome} {row.side_px:.0f}px" + (f" conf {row.conf:.2f}" if row.source == "pred" else "")
        cv2.rectangle(tile, (0, TILE - 20), (TILE, TILE), (255, 255, 255), -1)
        cv2.putText(tile, label, (4, TILE - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    per_row = per_row or len(tiles)
    blank = np.full_like(tiles[0], 255)
    tiles += [blank] * (-len(tiles) % per_row)
    rows = [np.hstack(tiles[k : k + per_row]) for k in range(0, len(tiles), per_row)]
    cv2.imwrite(str(path), np.vstack(rows))


def main() -> None:
    args = parse_args()
    matches = pd.read_csv(args.eval_dir / "matches.csv")

    by_size = recall_by_size(matches)
    by_size.round(4).to_csv(args.eval_dir / "recall_by_size.csv", lineterminator="\n")
    plot_recall_by_size(by_size, args.eval_dir / "recall_by_size.png")

    fp_causes = false_positive_causes(matches)
    fp = matches.loc[fp_causes.index].assign(cause=fp_causes)
    fn = matches[(matches.source == "gt") & (matches.outcome == "FN")].assign(cause=false_negative_causes(matches))
    causes = pd.concat([
        fp.groupby(["class_name", "cause"]).size().rename("count").reset_index().assign(error="FP"),
        fn.groupby(["class_name", "cause"]).size().rename("count").reset_index().assign(error="FN"),
    ])[["error", "class_name", "cause", "count"]]
    causes.to_csv(args.eval_dir / "error_causes.csv", index=False, lineterminator="\n")

    for name in ("helmet", "safety-vest"):
        missed = fn[fn.class_name == name].sort_values("side_px", ascending=False).head(args.examples)
        wrong = fp[fp.class_name == name].sort_values("conf", ascending=False).head(args.examples)
        example_grid(missed, matches, args.images, args.eval_dir / f"fn_{name}.jpg")
        example_grid(wrong, matches, args.images, args.eval_dir / f"fp_{name}.jpg")

    # Grade com todos os falsos positivos de capacete, em ordem de confiança, para a revisão visual.
    all_fp = fp[fp.class_name == "helmet"].sort_values("conf", ascending=False)
    example_grid(all_fp, matches, args.images, args.eval_dir / "fp_helmet_all.jpg", per_row=8)

    print("recall por tamanho:")
    print(by_size.round(3).to_string())
    print("\ncausas dos erros:")
    pivot = causes.pivot_table(index=["error", "class_name"], columns="cause", values="count", aggfunc="sum", fill_value=0)
    print(pivot.astype(int).to_string())
    print(f"\narquivos em {args.eval_dir}")


if __name__ == "__main__":
    main()
