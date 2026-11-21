import pandas as pd
import pytest

from epi.data.split import assign_groups, group_totals

FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}


def photographer_index() -> pd.DataFrame:
    # 20 fotógrafos com 3 imagens cada, todas com as quatro classes.
    return pd.DataFrame(
        {
            "photographer_id": [f"p{i // 3:02d}" for i in range(60)],
            "n_person": 2,
            "n_head": 1,
            "n_helmet": 1,
            "n_safety_vest": 1,
        }
    )


def test_each_photographer_goes_whole_to_one_split():
    index = photographer_index()
    split = assign_groups(group_totals(index, "photographer_id"), FRACTIONS)
    assert sorted(split.index) == sorted(index["photographer_id"].unique())
    assert set(split) <= set(FRACTIONS)


def test_split_follows_the_requested_fractions():
    index = photographer_index()
    split = assign_groups(group_totals(index, "photographer_id"), FRACTIONS)
    images = index["photographer_id"].map(split).value_counts(normalize=True)
    for name, fraction in FRACTIONS.items():
        assert images[name] == pytest.approx(fraction, abs=0.05)


def test_split_is_deterministic():
    totals = group_totals(photographer_index(), "photographer_id")
    assert assign_groups(totals, FRACTIONS).equals(assign_groups(totals, FRACTIONS))
