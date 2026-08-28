"""Task-agnostic modeling: an interpretable statistical baseline with full
regression diagnostics, a multi-model comparison, and automatic selection
of the best-performing candidate, for whichever task (regression or
binary/multiclass classification) the target implies.

The baseline mirrors the rigor of a classic OLS/Logit analysis: VIF-based
multicollinearity pruning, coefficient significance, and (for regression)
a Breusch-Pagan heteroscedasticity test plus residual/Q-Q data for plotting.
This is deliberately generalized from a single hardcoded housing-price OLS
into something that runs unmodified on any regression or classification
target.

The candidate model set (linear/regularized regression, random forest,
gradient boosting; logistic regression, random forest, gradient boosting)
mirrors the breadth of a full predictive-analytics course sequence —
OLS -> regularization -> trees -> ensembles — rather than committing to a
single algorithm.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF_THRESHOLD = 10.0
MAX_BASELINE_FEATURES = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200


@dataclass
class BaselineResult:
    kind: str  # "ols" | "logit" | "skipped"
    summary_text: str = ""
    pseudo_or_r_squared: float | None = None
    adj_r_squared: float | None = None
    coefficients: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    vif: dict[str, float] = field(default_factory=dict)
    dropped_for_multicollinearity: list[str] = field(default_factory=list)
    breusch_pagan_stat: float | None = None
    breusch_pagan_pvalue: float | None = None
    heteroscedastic: bool | None = None
    residual_mean: float | None = None
    fitted: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    note: str = ""


@dataclass
class ModelingResult:
    task: str
    target: str
    metrics: dict[str, float] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)
    confusion: list[list[int]] | None = None
    class_labels: list[str] | None = None
    baseline: BaselineResult | None = None
    n_train: int = 0
    n_test: int = 0
    model_comparison: dict[str, dict[str, float]] = field(default_factory=dict)
    best_model_name: str = ""


def _compute_vif(X: pd.DataFrame) -> pd.Series:
    X_const = sm.add_constant(X, has_constant="add")
    values = []
    with warnings.catch_warnings():
        # Near-perfect collinearity is expected mid-pruning (that's exactly what
        # we're iteratively removing) and shows up as ill-conditioning warnings.
        warnings.simplefilter("ignore")
        for i in range(1, X_const.shape[1]):
            try:
                values.append(variance_inflation_factor(X_const.values, i))
            except (ZeroDivisionError, np.linalg.LinAlgError):
                values.append(np.inf)
    return pd.Series(values, index=X.columns)


def _prune_by_vif(X: pd.DataFrame, threshold: float = VIF_THRESHOLD) -> tuple[pd.DataFrame, list[str]]:
    X = X.loc[:, X.std(numeric_only=True) > 1e-8]
    dropped: list[str] = []
    while X.shape[1] > 1:
        vif = _compute_vif(X).replace([np.inf, -np.inf], np.nan)
        if vif.isna().any():
            worst = vif[vif.isna()].index[0]
        elif vif.max() > threshold:
            worst = vif.idxmax()
        else:
            break
        X = X.drop(columns=[worst])
        dropped.append(worst)
    return X, dropped


def _select_baseline_columns(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    if X.shape[1] <= MAX_BASELINE_FEATURES:
        return X
    corr = X.apply(lambda col: abs(np.corrcoef(col, y)[0, 1]) if col.std() > 0 else 0.0)
    top = corr.sort_values(ascending=False).head(MAX_BASELINE_FEATURES).index
    return X[top]


def _fit_ols_baseline(X: pd.DataFrame, y: pd.Series) -> BaselineResult:
    X_sel = _select_baseline_columns(X, y)
    X_pruned, dropped = _prune_by_vif(X_sel)
    X_const = sm.add_constant(X_pruned)
    model = sm.OLS(y.values, X_const.values).fit()
    param_names = list(X_const.columns)
    resid = model.resid
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(resid, X_const.values)
    vif_final = _compute_vif(X_pruned)
    return BaselineResult(
        kind="ols",
        summary_text=str(model.summary()),
        pseudo_or_r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        coefficients=dict(zip(param_names, model.params.tolist())),
        p_values=dict(zip(param_names, model.pvalues.tolist())),
        vif=vif_final.to_dict(),
        dropped_for_multicollinearity=dropped,
        breusch_pagan_stat=float(bp_stat),
        breusch_pagan_pvalue=float(bp_pvalue),
        heteroscedastic=bool(bp_pvalue < 0.05),
        residual_mean=float(resid.mean()),
        fitted=model.fittedvalues.tolist(),
        residuals=resid.tolist(),
    )


def _fit_logit_baseline(X: pd.DataFrame, y: pd.Series) -> BaselineResult:
    X_sel = _select_baseline_columns(X, y)
    X_pruned, dropped = _prune_by_vif(X_sel)
    X_const = sm.add_constant(X_pruned)
    try:
        model = sm.Logit(y.values, X_const.values).fit(disp=0, maxiter=200)
    except Exception as exc:  # separation / non-convergence on small demo data
        return BaselineResult(kind="skipped", note=f"Logit baseline did not converge: {exc}")
    param_names = list(X_const.columns)
    return BaselineResult(
        kind="logit",
        summary_text=str(model.summary()),
        pseudo_or_r_squared=float(model.prsquared),
        coefficients=dict(zip(param_names, model.params.tolist())),
        p_values=dict(zip(param_names, model.pvalues.tolist())),
        vif=_compute_vif(X_pruned).to_dict(),
        dropped_for_multicollinearity=dropped,
    )


def _regression_candidates() -> dict[str, object]:
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge (CV)": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "Lasso (CV)": make_pipeline(
            StandardScaler(), LassoCV(alphas=np.logspace(-3, 2, 25), max_iter=20000, random_state=RANDOM_STATE)
        ),
        "Random Forest": RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def _classification_candidates() -> dict[str, object]:
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def run_modeling(df: pd.DataFrame, target: str, feature_cols: list[str], task: str) -> ModelingResult:
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_raw = df[target]

    if task == "classification":
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y_raw.astype(str)), index=df.index)
        class_labels = [str(c) for c in encoder.classes_]
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        class_labels = None

    stratify = y if (task == "classification" and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    result = ModelingResult(task=task, target=target, n_train=len(X_train), n_test=len(X_test))

    if task == "regression":
        result.baseline = _fit_ols_baseline(X_train, y_train)
        candidates = _regression_candidates()
        comparison: dict[str, dict[str, float]] = {}
        fitted_models: dict[str, object] = {}
        for name, model in candidates.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            comparison[name] = {
                "MAE": float(mean_absolute_error(y_test, preds)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
                "R2": float(r2_score(y_test, preds)),
            }
            fitted_models[name] = model
        best_name = max(comparison, key=lambda n: comparison[n]["R2"])
        scoring = "r2"
    else:
        n_classes = y.nunique()
        if n_classes == 2:
            result.baseline = _fit_logit_baseline(X_train, y_train)
        else:
            result.baseline = BaselineResult(
                kind="skipped", note="Interpretable baseline is limited to binary targets; skipped for multiclass."
            )
        candidates = _classification_candidates()
        comparison = {}
        fitted_models = {}
        for name, model in candidates.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            row = {
                "Accuracy": float(accuracy_score(y_test, preds)),
                "F1_weighted": float(f1_score(y_test, preds, average="weighted")),
            }
            if n_classes == 2:
                proba = model.predict_proba(X_test)[:, 1]
                row["ROC_AUC"] = float(roc_auc_score(y_test, proba))
            comparison[name] = row
            fitted_models[name] = model
        best_name = max(comparison, key=lambda n: comparison[n]["F1_weighted"])
        scoring = "f1_weighted"

    result.model_comparison = comparison
    result.best_model_name = best_name
    result.metrics = comparison[best_name]
    best_model = fitted_models[best_name]

    if task == "classification":
        best_preds = best_model.predict(X_test)
        result.confusion = confusion_matrix(y_test, best_preds).tolist()
        result.class_labels = class_labels

    importance = permutation_importance(
        best_model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, scoring=scoring
    )
    imp_series = pd.Series(importance.importances_mean, index=X.columns).sort_values(ascending=False)
    result.feature_importances = {k: float(v) for k, v in imp_series.head(15).items()}

    return result
