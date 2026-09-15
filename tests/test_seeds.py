"""`rewardhacking_training.seeds`: stable derivation and independent streams."""

from __future__ import annotations

from rewardhacking_training import seeds


def test_derive_seed_is_stable_and_in_range():
    assert seeds.derive_seed(0, "data/schedule") == seeds.derive_seed(0, "data/schedule")
    assert seeds.derive_seed(0, "data/schedule") == seeds.derive_seed("0/data/schedule")  # parts join with "/"
    assert 0 <= seeds.derive_seed(123, "x") < seeds.SEED_MODULUS
    assert seeds.derive_seed(0, "data/schedule") != seeds.derive_seed(1, "data/schedule")
    assert seeds.derive_seed(0, "data/schedule") != seeds.derive_seed(0, "data/nl_gameable")


def test_run_seeds_are_distinct_streams():
    s = seeds.run_seeds(0)
    assert set(s) == set(seeds.STREAMS) and len(set(s.values())) == len(seeds.STREAMS)
    assert seeds.run_seeds(0) == s and seeds.run_seeds(1) != s
