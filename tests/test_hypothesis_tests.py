import numpy as np
import pandas as pd

from tabular_autopilot.hypothesis_tests import (
    anova_tests,
    chi_square_tests,
    mann_whitney_tests,
    shapiro_wilk_test,
)


def test_chi_square_detects_a_real_association():
    rng = np.random.default_rng(0)
    n = 400
    group = rng.choice(["A", "B"], size=n)
    # Strong dependence: label follows group almost deterministically.
    label = np.where(group == "A", rng.choice([0, 1], size=n, p=[0.9, 0.1]), rng.choice([0, 1], size=n, p=[0.1, 0.9]))
    unrelated = rng.choice(["X", "Y", "Z"], size=n)  # independent of label
    df = pd.DataFrame({"group": group, "unrelated": unrelated, "label": label})

    results = chi_square_tests(df, ["group", "unrelated"], "label")
    by_feature = {r.feature: r for r in results}

    assert by_feature["group"].significant
    assert not by_feature["unrelated"].significant


def test_anova_detects_a_real_association_across_three_groups():
    rng = np.random.default_rng(1)
    n = 90
    label = rng.choice(["low", "mid", "high"], size=n)
    offset = pd.Series(label).map({"low": 0.0, "mid": 5.0, "high": 10.0}).to_numpy()
    signal = offset + rng.normal(scale=0.5, size=n)
    noise = rng.normal(size=n)
    df = pd.DataFrame({"signal": signal, "noise": noise, "label": label})

    results = anova_tests(df, ["signal", "noise"], "label")
    by_feature = {r.feature: r for r in results}

    assert by_feature["signal"].significant
    assert not by_feature["noise"].significant


def test_mann_whitney_detects_a_real_association_for_binary_target():
    rng = np.random.default_rng(2)
    n = 200
    label = rng.choice([0, 1], size=n)
    signal = np.where(label == 1, rng.normal(loc=5, size=n), rng.normal(loc=0, size=n))
    noise = rng.normal(size=n)
    df = pd.DataFrame({"signal": signal, "noise": noise, "label": label})

    results = mann_whitney_tests(df, ["signal", "noise"], "label")
    by_feature = {r.feature: r for r in results}

    assert by_feature["signal"].significant
    assert not by_feature["noise"].significant


def test_anova_skipped_for_binary_target():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "label": [0, 1, 0, 1]})
    assert anova_tests(df, ["x"], "label") == []


def test_mann_whitney_skipped_for_more_than_two_classes():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "label": ["a", "b", "c", "a", "b", "c"]})
    assert mann_whitney_tests(df, ["x"], "label") == []


def test_shapiro_wilk_flags_non_normal_residuals():
    rng = np.random.default_rng(3)
    skewed = rng.exponential(scale=1.0, size=500).tolist()
    result = shapiro_wilk_test(skewed)

    assert result is not None
    assert result.significant  # exponential residuals clearly aren't normal


def test_shapiro_wilk_none_for_too_few_points():
    assert shapiro_wilk_test([1.0, 2.0]) is None
