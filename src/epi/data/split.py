"""Divisão treino/validação/teste agrupada por fotógrafo e equilibrada por classe."""

from __future__ import annotations

import numpy as np
import pandas as pd

BALANCE_COLUMNS: tuple[str, ...] = ("images", "n_person", "n_head", "n_helmet", "n_safety_vest")


def group_totals(index: pd.DataFrame, group_column: str) -> pd.DataFrame:
    count_columns = [column for column in BALANCE_COLUMNS if column != "images"]
    totals = index.groupby(group_column)[count_columns].sum()
    totals.insert(0, "images", index.groupby(group_column).size())
    return totals


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
