import numpy as np
import pytest

from epi.track.ppe import HELMET_RULE, UNKNOWN, VEST_RULE, WITH, WITHOUT, Smoother, assign, ppe_state

# Pessoa de 100 x 200 px, com o capacete no topo e o colete no meio do corpo.
PERSON = np.array([[0.0, 0.0, 100.0, 200.0]])
HELMET = np.array([[30.0, 0.0, 70.0, 30.0]])
VEST = np.array([[20.0, 60.0, 80.0, 120.0]])


def test_assign_helmet_on_head_and_vest_on_torso():
    assert assign(HELMET, PERSON, HELMET_RULE).tolist() == [0]
    assert assign(VEST, PERSON, VEST_RULE).tolist() == [0]


def test_assign_rejects_item_outside_its_body_region():
    at_feet = np.array([[30.0, 170.0, 70.0, 200.0]])
    assert assign(at_feet, PERSON, HELMET_RULE).tolist() == [-1]
    assert assign(HELMET, PERSON, VEST_RULE).tolist() == [-1]


def test_assign_coverage_threshold():
    # 28 dos 40 px de largura dentro da pessoa: exatamente os 70% exigidos.
    at_threshold = np.array([[72.0, 0.0, 112.0, 30.0]])
    assert assign(at_threshold, PERSON, HELMET_RULE).tolist() == [0]
    half_outside = np.array([[80.0, 0.0, 120.0, 30.0]])
    assert assign(half_outside, PERSON, HELMET_RULE).tolist() == [-1]


def test_item_inside_two_persons_goes_to_only_one():
    persons = np.array([[0.0, 0.0, 100.0, 200.0], [60.0, 0.0, 160.0, 200.0]])
    shared = np.array([[65.0, 0.0, 95.0, 30.0]])
    assert assign(shared, persons, HELMET_RULE).tolist() == [0, -1]
    # O capacete disputado não impede a segunda pessoa de receber o próprio capacete.
    helmets = np.vstack([shared, [[110.0, 0.0, 150.0, 30.0]]])
    assert assign(helmets, persons, HELMET_RULE).tolist() == [0, 1]


def test_assign_without_items_or_persons():
    assert assign(np.empty((0, 4)), PERSON, HELMET_RULE).tolist() == [-1]
    assert assign(HELMET, np.empty((0, 4)), HELMET_RULE).tolist() == []


@pytest.mark.parametrize(
    ("found", "judgeable", "expected"),
    [(True, True, WITH), (True, False, WITH), (False, True, WITHOUT), (False, False, UNKNOWN)],
    ids=["found", "found-not-judgeable", "missing", "missing-not-judgeable"],
)
def test_ppe_state(found, judgeable, expected):
    assert ppe_state(found, judgeable) == expected


def test_smoother_waits_for_half_window_of_decisive_observations():
    smoother = Smoother(window=10, ratio=0.7)
    states = [smoother.update(WITH) for _ in range(5)]
    assert states == [UNKNOWN] * 4 + [WITH]


def test_smoother_does_not_count_unknown_observations():
    smoother = Smoother(window=10, ratio=0.7)
    for observation in [WITH, UNKNOWN] * 3 + [WITH]:
        state = smoother.update(observation)
    assert state == UNKNOWN  # 4 observações decisivas, menos que meia janela
    assert smoother.update(WITH) == WITH


def test_smoother_only_switches_when_the_ratio_agrees():
    smoother = Smoother(window=10, ratio=0.7)
    for _ in range(10):
        smoother.update(WITH)
    states = [smoother.update(WITHOUT) for _ in range(7)]
    # Com 4 a 6 "não" na janela, nenhum estado chega a 70% e o anterior se mantém.
    assert states == [WITH] * 6 + [WITHOUT]
