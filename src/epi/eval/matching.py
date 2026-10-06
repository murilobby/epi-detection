"""Casamento de detecções com anotações por IoU, para contar acertos e erros num limiar de confiança.

As caixas ficam em coordenadas normalizadas (x0, y0, x1, y1). Isso não altera o IoU: dentro de uma
imagem, todas as áreas são escaladas pelo mesmo fator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from epi.data.analysis import box_area, intersection_area


@dataclass(frozen=True)
class Boxes:
    xyxy: np.ndarray
    cls: np.ndarray
    conf: np.ndarray | None = None


@dataclass(frozen=True)
class ImageMatch:
    pred_to_gt: np.ndarray  # índice da anotação casada com cada detecção, ou -1
    gt_matched: np.ndarray  # se cada anotação foi encontrada por uma detecção da mesma classe
    confusions: list[tuple[int, int]]  # pares (detecção, anotação) de classes diferentes que se sobrepõem


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = intersection_area(a, b)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def greedy_match(iou: np.ndarray, order: np.ndarray, allowed: np.ndarray, threshold: float) -> np.ndarray:
    """Percorre as detecções em `order` e casa cada uma com a anotação livre e permitida de maior IoU."""
    pred_to_gt = np.full(iou.shape[0], -1)
    taken = np.zeros(iou.shape[1], dtype=bool)
    for i in order:
        candidates = allowed[i] & ~taken & (iou[i] >= threshold)
        if candidates.any():
            j = int(np.flatnonzero(candidates)[np.argmax(iou[i][candidates])])
            pred_to_gt[i] = j
            taken[j] = True
    return pred_to_gt


def match_image(pred: Boxes, gt: Boxes, iou_threshold: float) -> ImageMatch:
    iou = iou_matrix(pred.xyxy, gt.xyxy)
    order = np.argsort(-pred.conf, kind="stable")
    same_class = pred.cls[:, None] == gt.cls[None, :]
    pred_to_gt = greedy_match(iou, order, same_class, iou_threshold)
    gt_matched = np.zeros(len(gt.cls), dtype=bool)
    gt_matched[pred_to_gt[pred_to_gt >= 0]] = True

    # Entre o que sobrou, uma detecção sobre uma anotação de outra classe é confusão de classe, não fundo.
    leftover = (pred_to_gt < 0)[:, None] & ~gt_matched[None, :]
    cross = greedy_match(iou, order, leftover & ~same_class, iou_threshold)
    confusions = [(int(i), int(j)) for i, j in enumerate(cross) if j >= 0]
    return ImageMatch(pred_to_gt, gt_matched, confusions)


def update_confusion(matrix: np.ndarray, pred: Boxes, gt: Boxes, match: ImageMatch) -> None:
    """Linhas são a classe prevista e colunas a verdadeira; o último índice é o fundo."""
    background = matrix.shape[0] - 1
    confused_preds = {i for i, _ in match.confusions}
    confused_gts = {j for _, j in match.confusions}
    for i, j in enumerate(match.pred_to_gt):
        if j >= 0:
            matrix[pred.cls[i], gt.cls[j]] += 1
        elif i not in confused_preds:
            matrix[pred.cls[i], background] += 1
    for i, j in match.confusions:
        matrix[pred.cls[i], gt.cls[j]] += 1
    for j in np.flatnonzero(~match.gt_matched):
        if j not in confused_gts:
            matrix[background, gt.cls[j]] += 1
