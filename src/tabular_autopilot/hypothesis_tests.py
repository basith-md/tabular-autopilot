"""Formal hypothesis / association tests -- inferential statistics a data
scientist would reach for by hand, automated the same way every other stage
here is. These sit alongside the descriptive stats in ``profiling.py`` and
the model-comparison significance check in ``modeling.py``:

- **Chi-square test of independence**: each categorical feature vs a
  classification target.
- **One-way ANOVA**: each numeric feature vs a classification target with
  more than two classes.
- **Mann-Whitney U** (the non-parametric analogue of a two-sample t-test):
  each numeric feature vs a *binary* classification target, used instead of
  ANOVA specifically because ANOVA on exactly two groups reduces to a less
  robust t-test.
- **Shapiro-Wilk** normality test on the OLS baseline's residuals, run from
  ``modeling.py`` alongside the existing Breusch-Pagan heteroscedasticity
  check (both are regression-diagnostic tests on the same residual array).

Every test drops incomplete rows for the pair being tested (standard
practice for these tests) rather than requiring the whole frame to be
complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, mannwhitneyu, shapiro

ALPHA = 0.05
MAX_CATEGORIES_FOR_CHI_SQUARE = 20  # beyond this the contingency table is too sparse to mean much
SHAPIRO_MAX_SAMPLE = 5000


@dataclass
class AssociationTest:
    feature: str
    test_name: str
    statistic: float
    p_value: float
    significant: bool


@dataclass
class HypothesisTestSuite:
    """Classification-only bundle: chi-square for categorical features,
    and either ANOVA (>2 classes) or Mann-Whitney (exactly 2 classes) for
    numeric ones -- never both, since ANOVA on two groups just reduces to a
    less robust version of the same question Mann-Whitney answers."""

    chi_square: list[AssociationTest] = field(default_factory=list)
    anova: list[AssociationTest] = field(default_factory=list)
    mann_whitney: list[AssociationTest] = field(default_factory=list)


def chi_square_tests(df: pd.DataFrame, categorical_cols: list[str], target_col: str) -> list[AssociationTest]:
    if target_col not in df.columns:
        return []
    results: list[AssociationTest] = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        subset = df[[col, target_col]].dropna()
        if not (2 <= subset[col].nunique() <= MAX_CATEGORIES_FOR_CHI_SQUARE):
            continue
        table = pd.crosstab(subset[col], subset[target_col])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        try:
            stat, p, _, _ = chi2_contingency(table)
        except ValueError:
            continue
        results.append(AssociationTest(col, "Chi-square", float(stat), float(p), bool(p < ALPHA)))
    return sorted(results, key=lambda r: r.p_value)


def anova_tests(df: pd.DataFrame, numeric_cols: list[str], target_col: str) -> list[AssociationTest]:
    if target_col not in df.columns or df[target_col].dropna().nunique() < 3:
        return []
    results: list[AssociationTest] = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        subset = df[[col, target_col]].dropna()
        groups = [g[col].to_numpy() for _, g in subset.groupby(target_col) if len(g) >= 2]
        if len(groups) < 2:
            continue
        try:
            stat, p = f_oneway(*groups)
        except ValueError:
            continue
        if stat != stat or p != p:  # NaN guard (e.g. a zero-variance group)
            continue
        results.append(AssociationTest(col, "ANOVA F-test", float(stat), float(p), bool(p < ALPHA)))
    return sorted(results, key=lambda r: r.p_value)


def mann_whitney_tests(df: pd.DataFrame, numeric_cols: list[str], target_col: str) -> list[AssociationTest]:
    if target_col not in df.columns:
        return []
    classes = df[target_col].dropna().unique()
    if len(classes) != 2:
        return []
    label_a, label_b = classes[0], classes[1]
    results: list[AssociationTest] = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        subset = df[[col, target_col]].dropna()
        a = subset.loc[subset[target_col] == label_a, col]
        b = subset.loc[subset[target_col] == label_b, col]
        if len(a) < 2 or len(b) < 2:
            continue
        try:
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        results.append(AssociationTest(col, "Mann-Whitney U", float(stat), float(p), bool(p < ALPHA)))
    return sorted(results, key=lambda r: r.p_value)


def shapiro_wilk_test(residuals: list[float]) -> AssociationTest | None:
    if len(residuals) < 3:
        return None
    sample = residuals[:SHAPIRO_MAX_SAMPLE]
    try:
        stat, p = shapiro(sample)
    except ValueError:
        return None
    return AssociationTest("residuals", "Shapiro-Wilk", float(stat), float(p), bool(p < ALPHA))
