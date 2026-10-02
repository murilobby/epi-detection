"""Estatísticas do dataset convertido: classes, tamanho das caixas, relações entre caixas e fontes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from epi.data.convert import CLASSES

SPLITS: tuple[str, ...] = ("train", "val", "test")
# Limites do COCO para objeto pequeno e médio, como lado equivalente (raiz da área) em pixels.
SMALL_MAX, MEDIUM_MAX = 32, 96


def load_tables(dataset_dir: Path, split_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = pd.read_csv(split_csv)[["stem", "split"]]
    index = pd.read_csv(dataset_dir / "index.csv").merge(split, on="stem", validate="one_to_one")
    boxes = pd.read_csv(dataset_dir / "boxes.csv").merge(
        index[["stem", "split", "width", "height"]], on="stem", validate="many_to_one"
    )
    return index, boxes


def class_summary(boxes: pd.DataFrame) -> pd.DataFrame:
    instances = pd.crosstab(boxes["class_name"], boxes["split"]).reindex(index=CLASSES, columns=SPLITS)
    table = instances.add_prefix("instances_").rename_axis(columns=None)
    table.insert(0, "instances", instances.sum(axis=1))
    table.insert(1, "images_with_class", boxes.groupby("class_name")["stem"].nunique().reindex(CLASSES))
    table["share_pct"] = (table["instances"] / table["instances"].sum() * 100).round(1)
    table["largest_to_class_ratio"] = (table["instances"].max() / table["instances"]).round(1)
    return table


def image_summary(index: pd.DataFrame) -> pd.DataFrame:
    count_columns = [column for column in index.columns if column.startswith("n_")]
    boxes_per_image = index[count_columns].sum(axis=1)
    grouped = boxes_per_image.groupby(index["split"])
    table = pd.DataFrame(
        {
            "images": grouped.size(),
            "background_images": grouped.apply(lambda counts: int((counts == 0).sum())),
            "mean_boxes_per_image": grouped.mean().round(2),
            "max_boxes_per_image": grouped.max(),
        }
    ).reindex(SPLITS)
    table["background_pct"] = (table["background_images"] / table["images"] * 100).round(1)
    return table


def pixel_side(boxes: pd.DataFrame, imgsz: int) -> pd.Series:
    """Lado equivalente de cada caixa, em pixels, com o lado maior da imagem redimensionado para imgsz."""
    scale = imgsz / boxes[["width", "height"]].max(axis=1)
    return np.sqrt(boxes["w"] * boxes["width"] * boxes["h"] * boxes["height"]) * scale


def size_buckets(boxes: pd.DataFrame, imgsz: int) -> pd.DataFrame:
    side = pixel_side(boxes, imgsz)
    bucket = pd.cut(
        side, [0, SMALL_MAX, MEDIUM_MAX, np.inf], right=False, labels=["small", "medium", "large"]
    )
    table = (pd.crosstab(boxes["class_name"], bucket, normalize="index") * 100).reindex(CLASSES)
    table = table.add_suffix("_pct").rename_axis(columns=None)
    table["median_side_px"] = side.groupby(boxes["class_name"]).median().reindex(CLASSES)
    table.insert(0, "imgsz", imgsz)
    return table.round(1)


def to_xyxy(frame: pd.DataFrame) -> np.ndarray:
    half_w, half_h = frame["w"] / 2, frame["h"] / 2
    return np.column_stack(
        [frame["cx"] - half_w, frame["cy"] - half_h, frame["cx"] + half_w, frame["cy"] + half_h]
    )


def intersection_area(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Área de interseção entre cada caixa de a e cada caixa de b, em formato (len(a), len(b))."""
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)


def box_area(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def max_overlap(
    boxes: pd.DataFrame, source: str, target: str, normalize_by: Literal["source", "target"]
) -> np.ndarray:
    """Para cada caixa source, a maior interseção com uma caixa target da mesma imagem.

    A interseção é dividida pela área da caixa source ou da target. Coordenadas normalizadas
    bastam: dentro de uma imagem, as duas áreas são escaladas pelo mesmo fator.
    """
    values = []
    for _, image in boxes[boxes["class_name"].isin([source, target])].groupby("stem"):
        src = to_xyxy(image[image["class_name"] == source])
        if len(src) == 0:
            continue
        tgt = to_xyxy(image[image["class_name"] == target])
        if len(tgt) == 0:
            values.append(np.zeros(len(src)))
            continue
        inter = intersection_area(src, tgt)
        denominator = box_area(src)[:, None] if normalize_by == "source" else box_area(tgt)[None, :]
        values.append((inter / denominator).max(axis=1))
    return np.concatenate(values)


def box_relations(boxes: pd.DataFrame) -> pd.DataFrame:
    """Percentual de caixas que atendem a cada relação espacial; base para associar EPI a pessoa."""
    checks = [
        ("helmet", "person", "source", 0.9, "capacete com >= 90% da área dentro de uma pessoa"),
        ("head", "person", "source", 0.9, "cabeça com >= 90% da área dentro de uma pessoa"),
        ("safety-vest", "person", "source", 0.9, "colete com >= 90% da área dentro de uma pessoa"),
        ("helmet", "head", "source", 0.5, "capacete com >= 50% da área dentro de uma cabeça"),
        ("head", "helmet", "target", 0.5, "cabeça que contém >= 50% de algum capacete"),
    ]
    rows = []
    for source, target, normalize_by, threshold, description in checks:
        overlap = max_overlap(boxes, source, target, normalize_by)
        rows.append(
            {
                "relation": description,
                "boxes": len(overlap),
                "pct": round(float((overlap >= threshold).mean() * 100), 1),
            }
        )
    return pd.DataFrame(rows)


def photographer_concentration(index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        counts = index.loc[index["split"] == split, "photographer_id"].value_counts()
        rows.append(
            {
                "split": split,
                "photographers": len(counts),
                "images": int(counts.sum()),
                "top1_pct": counts.iloc[0] / counts.sum() * 100,
                "top10_pct": counts.iloc[:10].sum() / counts.sum() * 100,
                "median_images_per_photographer": counts.median(),
            }
        )
    return pd.DataFrame(rows).set_index("split").round(1)


def concentration_curve(index: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Curva de Lorenz invertida: % de fotógrafos (do maior para o menor) contra % de imagens."""
    counts = index.loc[index["split"] == split, "photographer_id"].value_counts().to_numpy()
    x = np.arange(len(counts) + 1) / len(counts) * 100
    y = np.concatenate([[0], np.cumsum(counts) / counts.sum() * 100])
    return x, y
