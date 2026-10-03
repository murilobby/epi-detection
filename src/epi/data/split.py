"""Divisão treino/validação/teste agrupada por fotógrafo e equilibrada por classe."""

from __future__ import annotations

import numpy as np
import pandas as pd

COUNT_COLUMNS: tuple[str, ...] = ("n_person", "n_head", "n_helmet", "n_safety_vest")
PRESENCE_COLUMNS: tuple[str, ...] = tuple(f"img_{column[2:]}" for column in COUNT_COLUMNS)
BALANCE_COLUMNS: tuple[str, ...] = ("images", *COUNT_COLUMNS, *PRESENCE_COLUMNS)


def balance_table(index: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por imagem com o que deve ficar equilibrado: caixas por classe e presença de cada classe.

    Equilibrar só caixas deixaria um conjunto com poucas fotos de multidão concentrando uma classe rara.
    """
    table = index[list(COUNT_COLUMNS)].copy()
    table.insert(0, "images", 1)
    for count, presence in zip(COUNT_COLUMNS, PRESENCE_COLUMNS):
        table[presence] = (index[count] > 0).astype(int)
    return table


def group_totals(index: pd.DataFrame, group_column: str) -> pd.DataFrame:
    return balance_table(index).groupby(index[group_column]).sum()


def split_cost(split_totals: np.ndarray, grand_total: np.ndarray, fractions: np.ndarray) -> float:
    """Soma, sobre conjuntos e colunas, do erro quadrático entre a fração obtida e a desejada."""
    obtained = split_totals / grand_total
    return float(((obtained - fractions[:, None]) ** 2).sum())


def assign_groups(totals: pd.DataFrame, fractions: dict[str, float]) -> pd.Series:
    """Atribui cada grupo inteiro ao conjunto que mais reduz o desequilíbrio (guloso, determinístico)."""
    names = list(fractions)
    target = np.array([fractions[name] for name in names])
    values = totals[list(BALANCE_COLUMNS)].to_numpy(dtype=float)
    grand_total = values.sum(axis=0)

    # Grupos grandes primeiro: são os mais difíceis de encaixar; os pequenos ajustam no final.
    order = sorted(range(len(totals)), key=lambda i: (-values[i, 0], totals.index[i]))

    split_totals = np.zeros((len(names), values.shape[1]))
    assignment = {}
    for i in order:
        costs = []
        for s in range(len(names)):
            split_totals[s] += values[i]
            costs.append(split_cost(split_totals, grand_total, target))
            split_totals[s] -= values[i]
        best = int(np.argmin(costs))
        split_totals[best] += values[i]
        assignment[totals.index[i]] = names[best]
    return pd.Series(assignment, name="split")
