"""Classificação dos erros do detector a partir da tabela de casamentos gerada na avaliação."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epi.data.analysis import MEDIUM_MAX, SMALL_MAX
from epi.eval.matching import iou_matrix

SIZE_LABELS: tuple[str, ...] = ("pequeno", "médio", "grande")
XYXY = ["x0", "y0", "x1", "y1"]


def size_bucket(side: pd.Series) -> pd.Series:
    return pd.cut(side, [0, SMALL_MAX, MEDIUM_MAX, np.inf], right=False, labels=SIZE_LABELS)


def recall_by_size(matches: pd.DataFrame) -> pd.DataFrame:
    gt = matches[matches.source == "gt"]
    found = (gt.outcome == "TP").groupby([gt.class_name, size_bucket(gt.side_px)], observed=False)
    table = pd.DataFrame({"instances": found.size(), "found": found.sum()})
    table["recall"] = table.found / table.instances.replace(0, np.nan)
    return table


def false_positive_causes(matches: pd.DataFrame) -> pd.Series:
    """Causa de cada falso positivo, em ordem de prioridade: duplicata, classe trocada, localização, fundo."""
    causes = {}
    for _, image in matches.groupby("stem"):
        fp = image[(image.source == "pred") & (image.outcome == "FP")]
        if fp.empty:
            continue
        gt = image[image.source == "gt"]
        iou = iou_matrix(fp[XYXY].to_numpy(), gt[XYXY].to_numpy())
        same_class = fp.class_name.to_numpy()[:, None] == gt.class_name.to_numpy()[None, :]
        best_same = np.where(same_class, iou, 0.0).max(axis=1, initial=0.0)
        for k, (index, row) in enumerate(fp.iterrows()):
            # IoU >= 0,5 com uma anotação da mesma classe só sobra como erro se ela já foi casada antes.
            if best_same[k] >= 0.5:
                causes[index] = "duplicata"
            elif isinstance(row.confused_with, str):
                causes[index] = "classe trocada"
            elif best_same[k] >= 0.1:
                causes[index] = "localização"
            else:
                causes[index] = "fundo"
    return pd.Series(causes, name="cause", dtype="object")


def false_negative_causes(matches: pd.DataFrame) -> pd.Series:
    fn = matches[(matches.source == "gt") & (matches.outcome == "FN")]
    return fn.confused_with.map(lambda other: "classe trocada" if isinstance(other, str) else "não detectado")
