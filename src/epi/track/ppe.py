"""Associação de cabeça, capacete e colete a cada pessoa rastreada, e suavização do estado no tempo.

As regras vêm da análise do dataset: 95% das cabeças e 88% dos capacetes e coletes anotados ficam quase
inteiros dentro da caixa da pessoa. Foram fixadas antes de rodar nos vídeos de demonstração.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from epi.data.analysis import box_area, intersection_area

WITH, WITHOUT, UNKNOWN = "sim", "não", "indefinido"


@dataclass(frozen=True)
class Rule:
    min_coverage: float  # fração mínima da caixa do item dentro da caixa da pessoa
    y_range: tuple[float, float]  # faixa do centro do item, de 0 (topo) a 1 (base) da pessoa


HEAD_RULE = Rule(0.7, (-0.05, 0.4))
HELMET_RULE = Rule(0.7, (-0.05, 0.4))
VEST_RULE = Rule(0.7, (0.15, 0.75))


def coverage(items: np.ndarray, persons: np.ndarray) -> np.ndarray:
    """Fração da área de cada item que fica dentro de cada pessoa, em formato (itens, pessoas)."""
    return intersection_area(items, persons) / np.maximum(box_area(items)[:, None], 1e-9)


def relative_center_y(items: np.ndarray, persons: np.ndarray) -> np.ndarray:
    center = (items[:, 1] + items[:, 3]) / 2
    height = np.maximum(persons[:, 3] - persons[:, 1], 1e-9)
    return (center[:, None] - persons[None, :, 1]) / height[None, :]


def assign(items: np.ndarray, persons: np.ndarray, rule: Rule) -> np.ndarray:
    """Índice do item atribuído a cada pessoa, ou -1; cada item vai para no máximo uma pessoa."""
    result = np.full(len(persons), -1)
    if len(items) == 0 or len(persons) == 0:
        return result
    cov = coverage(items, persons)
    y = relative_center_y(items, persons)
    valid = (cov >= rule.min_coverage) & (y >= rule.y_range[0]) & (y <= rule.y_range[1])
    score = np.where(valid, cov, 0.0)
    used = set()
    # Guloso pela maior cobertura: decide itens que caem dentro de duas pessoas sobrepostas.
    for flat in np.argsort(-score, axis=None, kind="stable"):
        item, person = np.unravel_index(flat, score.shape)
        if score[item, person] == 0:
            break
        if result[person] < 0 and item not in used:
            result[person] = item
            used.add(item)
    return result


def ppe_state(found: bool, judgeable: bool) -> str:
    """"sim" se o EPI foi detectado; "não" só se a pessoa puder ser julgada; senão "indefinido".

    Uma pessoa pode ser julgada quando tem cabeça detectada e grande o bastante para o detector
    enxergar o EPI; caso contrário, a ausência de detecção não prova a ausência do EPI.
    """
    if found:
        return WITH
    return WITHOUT if judgeable else UNKNOWN


@dataclass
class Smoother:
    """Só troca de estado quando uma fração `ratio` das observações decisivas da janela concorda."""

    window: int
    ratio: float
    state: str = UNKNOWN
    history: deque = field(init=False)

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.window)

    def update(self, observation: str) -> str:
        self.history.append(observation)
        decisive = [s for s in self.history if s != UNKNOWN]
        # Exige ao menos meia janela de observações decisivas antes de decidir qualquer coisa.
        if len(decisive) >= self.window // 2:
            for candidate in (WITH, WITHOUT):
                if decisive.count(candidate) >= self.ratio * len(decisive):
                    self.state = candidate
        return self.state
