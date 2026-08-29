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

Three further, independently-togglable pieces of rigor:
- **Feature selection** (variance threshold, then SelectKBest) so a wide
  feature matrix (e.g. after TF-IDF text vectorization) doesn't drown the
  model comparison in noise columns.
- **Class-imbalance handling** for classification: detected automatically,
  and — when enabled — addressed via class_weight="balanced" for models
  that support it as a constructor argument, or sample_weight at fit time
  for the one that doesn't (HistGradientBoostingClassifier).
- **Cross-validation** as an alternative to a single train/test split for
  the headline comparison metrics, opt-in since it multiplies runtime by
  the fold count.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wilcoxon
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from tabular_autopilot.hypothesis_tests import AssociationTest, shapiro_wilk_test

VIF_THRESHOLD = 10.0
MAX_BASELINE_FEATURES = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
TREE_BASED_MODELS = {"Random Forest", "Gradient Boosting"}
ILLUSTRATIVE_TREE_MAX_DEPTH = 3
IMBALANCE_RATIO_THRESHOLD = 1.5
MAX_SELECTED_FEATURES = 50
NEAR_ZERO_VARIANCE_THRESHOLD = 1e-8


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
    normality_test: AssociationTest | None = None
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
    is_tree_based: bool = False
    illustrative_tree: object | None = None
    illustrative_tree_features: list[str] = field(default_factory=list)
    is_imbalanced: bool = False
    class_weight_applied: bool = False
    feature_selection_applied: bool = False
    dropped_low_variance_features: list[str] = field(default_factory=list)
    n_features_before_selection: int = 0
    n_features_after_selection: int = 0
    cv_folds: int = 0
    hyperparameter_search_applied: bool = False
    broad_hyperparameter_search_applied: bool = False
    best_hyperparameters: dict[str, object] = field(default_factory=dict)
    target_encoded_features: list[str] = field(default_factory=list)
    runner_up_model_name: str = ""
    best_vs_runner_up_test: AssociationTest | None = None


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
        normality_test=shapiro_wilk_test(resid.tolist()),
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


def _regression_candidates(
    ridge_alpha_range: tuple[float, float] = (1e-3, 1e3),
    lasso_alpha_range: tuple[float, float] = (1e-3, 1e2),
    rf_n_estimators: int = N_ESTIMATORS,
    rf_max_depth: int | None = None,
    gb_learning_rate: float = 0.1,
    gb_max_iter: int = 100,
) -> dict[str, object]:
    ridge_lo, ridge_hi = ridge_alpha_range
    lasso_lo, lasso_hi = lasso_alpha_range
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge (CV)": make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(np.log10(ridge_lo), np.log10(ridge_hi), 25))
        ),
        "Lasso (CV)": make_pipeline(
            StandardScaler(),
            LassoCV(
                alphas=np.logspace(np.log10(lasso_lo), np.log10(lasso_hi), 25),
                max_iter=20000,
                random_state=RANDOM_STATE,
            ),
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            learning_rate=gb_learning_rate, max_iter=gb_max_iter, random_state=RANDOM_STATE
        ),
    }


def _classification_candidates(
    class_weight: str | None = None,
    logreg_C: float = 1.0,
    rf_n_estimators: int = N_ESTIMATORS,
    rf_max_depth: int | None = None,
    gb_learning_rate: float = 0.1,
    gb_max_iter: int = 100,
) -> dict[str, object]:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight=class_weight, C=logreg_C)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=class_weight,
        ),
        # HistGradientBoostingClassifier has no class_weight constructor arg;
        # imbalance is instead handled via sample_weight at fit time (below).
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=gb_learning_rate, max_iter=gb_max_iter, random_state=RANDOM_STATE
        ),
    }


def _param_grid_for(
    name: str,
    rf_n_estimators: int,
    rf_max_depth: int | None,
    gb_learning_rate: float,
    gb_max_iter: int,
    logreg_C: float,
) -> dict[str, list] | None:
    """The one grid-search-able knob per model -- the lever a practitioner
    reaches for first -- centered on whatever value was configured, so
    turning search on refines around the user's own setting rather than a
    fixed, disconnected default grid."""
    if name == "Random Forest":
        return {
            "n_estimators": list(dict.fromkeys([max(50, rf_n_estimators // 2), rf_n_estimators, rf_n_estimators * 2])),
            "max_depth": list(dict.fromkeys([rf_max_depth, 8, 16])),
        }
    if name == "Gradient Boosting":
        return {
            "learning_rate": list(dict.fromkeys([0.03, gb_learning_rate, 0.3])),
            "max_iter": list(dict.fromkeys([50, gb_max_iter, 200])),
        }
    if name == "Logistic Regression":
        return {"logisticregression__C": list(dict.fromkeys([0.1, logreg_C, 10.0]))}
    return None


def _broad_param_distributions_for(
    name: str,
    rf_n_estimators: int,
    rf_max_depth: int | None,
    gb_learning_rate: float,
    gb_max_iter: int,
    logreg_C: float,
) -> dict[str, list] | None:
    """A wider distribution per model for RandomizedSearchCV, used instead of
    ``_param_grid_for``'s small 2-3 value grid when the broader search is
    requested -- still centered on the configured values, just with more
    room around them, and sampled rather than exhaustively evaluated so the
    fit count stays bounded regardless of how wide the ranges are."""
    if name == "Random Forest":
        return {
            "n_estimators": sorted({50, 100, rf_n_estimators, 300, 400, 500}),
            "max_depth": [rf_max_depth, 4, 8, 12, 16, 24],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", None],
        }
    if name == "Gradient Boosting":
        return {
            "learning_rate": sorted({0.01, 0.03, 0.05, gb_learning_rate, 0.2, 0.3}),
            "max_iter": sorted({50, gb_max_iter, 150, 200, 300}),
            "max_depth": [None, 3, 5, 8],
            "max_leaf_nodes": [15, 31, 63],
        }
    if name == "Logistic Regression":
        return {"logisticregression__C": sorted({0.01, 0.03, 0.1, logreg_C, 3.0, 10.0, 30.0})}
    return None


def _target_encode_column(
    train_col: pd.Series, y_train: pd.Series, test_col: pd.Series, n_folds: int = 5, seed: int = RANDOM_STATE
) -> tuple[pd.Series, pd.Series]:
    """Out-of-fold mean-target encoding: each training row's encoded value
    comes from a fold it wasn't used to compute, so the model can't just
    memorize "this exact category had this exact target." The test split
    uses the full training-set mapping, since none of it can leak into test
    rows the way it could leak between training rows."""
    y_numeric = pd.to_numeric(y_train, errors="coerce")
    global_mean = float(y_numeric.mean())
    folds = max(2, min(n_folds, train_col.shape[0]))
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    oof = pd.Series(np.nan, index=train_col.index, dtype=float)
    for train_idx, val_idx in kf.split(train_col):
        fold_means = y_numeric.iloc[train_idx].groupby(train_col.iloc[train_idx]).mean()
        oof.iloc[val_idx] = train_col.iloc[val_idx].map(fold_means).to_numpy()
    oof = oof.fillna(global_mean)

    full_means = y_numeric.groupby(train_col).mean()
    test_encoded = test_col.map(full_means).fillna(global_mean)
    return oof, test_encoded


def _extract_fitted_hyperparameters(name: str, fitted_model: object) -> dict[str, object]:
    """What the winning model actually ran with -- the alpha RidgeCV/LassoCV
    picked via their own internal search (always), or the search winner for
    the other models (only when grid or randomized search was enabled)."""
    is_grid = isinstance(fitted_model, (GridSearchCV, RandomizedSearchCV))
    try:
        if name == "Ridge (CV)":
            return {"alpha": round(float(fitted_model.named_steps["ridgecv"].alpha_), 6)}
        if name == "Lasso (CV)":
            return {"alpha": round(float(fitted_model.named_steps["lassocv"].alpha_), 6)}
        if name == "Logistic Regression":
            if is_grid:
                return {"C": float(fitted_model.best_params_["logisticregression__C"])}
            return {"C": float(fitted_model.named_steps["logisticregression"].C)}
        if name == "Random Forest":
            if is_grid:
                return dict(fitted_model.best_params_)
            return {"n_estimators": fitted_model.n_estimators, "max_depth": fitted_model.max_depth}
        if name == "Gradient Boosting":
            if is_grid:
                return dict(fitted_model.best_params_)
            return {"learning_rate": fitted_model.learning_rate, "max_iter": fitted_model.max_iter}
    except (AttributeError, KeyError):
        return {}
    return {}


def available_model_names() -> list[str]:
    """Union of every candidate name across both tasks, in a stable display
    order -- used to populate a "which models to compare" UI control before
    the task (and therefore the applicable subset) is known."""
    seen: dict[str, None] = {}
    for name in list(_regression_candidates()) + list(_classification_candidates()):
        seen.setdefault(name, None)
    return list(seen)


def _filter_candidates(candidates: dict[str, object], model_names: list[str] | None) -> dict[str, object]:
    if not model_names:
        return candidates
    filtered = {name: model for name, model in candidates.items() if name in model_names}
    return filtered or candidates  # never leave the comparison empty


def _select_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, task: str, max_features: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop near-zero-variance columns, then (if still too wide) keep only
    the top-scoring ``max_features`` by an ANOVA F-test against the target.
    Fit on the training split only -- the test split is just re-indexed to
    match, so no information from it leaks into which features are kept."""
    dropped_low_variance: list[str] = []
    variances = X_train.var(numeric_only=True)
    keep_cols = variances[variances > NEAR_ZERO_VARIANCE_THRESHOLD].index.tolist()
    if len(keep_cols) < X_train.shape[1]:
        dropped_low_variance = [c for c in X_train.columns if c not in keep_cols]
        X_train, X_test = X_train[keep_cols], X_test[keep_cols]

    if X_train.shape[1] > max_features:
        score_func = f_classif if task == "classification" else f_regression
        selector = SelectKBest(score_func=score_func, k=max_features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # constant-column / low-count warnings from the score function
            selector.fit(X_train, y_train)
        keep_mask = selector.get_support()
        X_train, X_test = X_train.loc[:, keep_mask], X_test.loc[:, keep_mask]

    return X_train, X_test, dropped_low_variance


def _cv_scoring_spec(task: str, n_classes: int | None) -> dict[str, tuple[str, int]]:
    """Maps our display metric name -> (sklearn scorer string, sign). Sign is
    -1 for sklearn's "neg_*" scorers so the reported number matches the
    single-split path's convention (positive MAE/RMSE, not negative)."""
    if task == "regression":
        return {
            "MAE": ("neg_mean_absolute_error", -1),
            "RMSE": ("neg_root_mean_squared_error", -1),
            "R2": ("r2", 1),
        }
    spec = {
        "Accuracy": ("accuracy", 1),
        "F1_weighted": ("f1_weighted", 1),
        "Balanced_Accuracy": ("balanced_accuracy", 1),
    }
    if n_classes == 2:
        spec["ROC_AUC"] = ("roc_auc", 1)
    return spec


def _cross_validate_metrics(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    scoring_spec: dict[str, tuple[str, int]],
    primary_metric: str,
) -> tuple[dict[str, float], np.ndarray]:
    scoring = {display: sk_name for display, (sk_name, _sign) in scoring_spec.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(model, X, y, cv=cv_folds, scoring=scoring)
    metrics = {
        display: float(sign * np.mean(scores[f"test_{display}"])) for display, (_sk, sign) in scoring_spec.items()
    }
    sign = scoring_spec[primary_metric][1]
    primary_folds = sign * scores[f"test_{primary_metric}"]
    return metrics, primary_folds


def run_modeling(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    task: str,
    test_size: float = TEST_SIZE,
    model_names: list[str] | None = None,
    handle_imbalance: bool = True,
    feature_selection: bool = True,
    cv_folds: int = 0,
    ridge_alpha_range: tuple[float, float] = (1e-3, 1e3),
    lasso_alpha_range: tuple[float, float] = (1e-3, 1e2),
    logreg_C: float = 1.0,
    rf_n_estimators: int = N_ESTIMATORS,
    rf_max_depth: int | None = None,
    gb_learning_rate: float = 0.1,
    gb_max_iter: int = 100,
    hyperparameter_search: bool = False,
    broad_hyperparameter_search: bool = False,
    target_encode_cols: list[str] | None = None,
) -> ModelingResult:
    X_raw = df[feature_cols].copy()
    y_raw = df[target]

    if task == "classification":
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y_raw.astype(str)), index=df.index)
        class_labels = [str(c) for c in encoder.classes_]
        n_classes = int(y.nunique())
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        class_labels = None
        n_classes = None

    stratify = y if (task == "classification" and y.value_counts().min() >= 2) else None
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
    )

    # Target encoding needs y_train, so it has to happen after the split --
    # fit on the training fold only (with an out-of-fold pass within it) and
    # apply that mapping to the test fold, the same leakage discipline as
    # everything else here.
    target_encode_cols = [c for c in (target_encode_cols or []) if c in X_train_raw.columns]
    X_train, X_test = X_train_raw.copy(), X_test_raw.copy()
    for col in target_encode_cols:
        X_train[col], X_test[col] = _target_encode_column(X_train_raw[col], y_train, X_test_raw[col])
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    result = ModelingResult(
        task=task, target=target, n_train=len(X_train), n_test=len(X_test), target_encoded_features=target_encode_cols
    )

    n_features_before = X_train.shape[1]
    dropped_low_variance: list[str] = []
    if feature_selection:
        X_train, X_test, dropped_low_variance = _select_features(
            X_train, X_test, y_train, task, MAX_SELECTED_FEATURES
        )
    result.feature_selection_applied = feature_selection
    result.dropped_low_variance_features = dropped_low_variance
    result.n_features_before_selection = n_features_before
    result.n_features_after_selection = X_train.shape[1]

    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])

    class_weight = None
    sample_weight = None
    is_imbalanced = False
    if task == "classification":
        counts = y_train.value_counts()
        ratio = float(counts.max() / counts.min()) if counts.min() else float("inf")
        is_imbalanced = ratio >= IMBALANCE_RATIO_THRESHOLD
        if handle_imbalance and is_imbalanced:
            class_weight = "balanced"
            sample_weight = compute_sample_weight("balanced", y_train)
    result.is_imbalanced = is_imbalanced
    result.class_weight_applied = class_weight is not None

    use_cv = cv_folds and cv_folds >= 2
    if use_cv and task == "classification":
        cv_folds = max(2, min(cv_folds, int(y_full.value_counts().min())))
        use_cv = cv_folds >= 2
    result.cv_folds = cv_folds if use_cv else 0

    if task == "regression":
        result.baseline = _fit_ols_baseline(X_train, y_train)
        candidates = _filter_candidates(
            _regression_candidates(
                ridge_alpha_range, lasso_alpha_range, rf_n_estimators, rf_max_depth, gb_learning_rate, gb_max_iter
            ),
            model_names,
        )
        primary_metric = "R2"
    else:
        if n_classes == 2:
            result.baseline = _fit_logit_baseline(X_train, y_train)
        else:
            result.baseline = BaselineResult(
                kind="skipped", note="Interpretable baseline is limited to binary targets; skipped for multiclass."
            )
        classification_candidates = _classification_candidates(
            class_weight, logreg_C, rf_n_estimators, rf_max_depth, gb_learning_rate, gb_max_iter
        )
        candidates = _filter_candidates(classification_candidates, model_names)
        primary_metric = "F1_weighted"

    # Nested CV (search inside k-fold CV) multiplies runtime by the search
    # size on top of the fold count -- too slow for the in-browser demo, so
    # search only applies to the single-split path.
    do_grid_search = bool(hyperparameter_search) and not use_cv
    do_broad_search = do_grid_search and bool(broad_hyperparameter_search)
    if do_grid_search:
        param_source = _broad_param_distributions_for if do_broad_search else _param_grid_for
        candidates = {
            name: (
                RandomizedSearchCV(model, grid, cv=3, n_iter=10, random_state=RANDOM_STATE)
                if do_broad_search and grid is not None
                else (GridSearchCV(model, grid, cv=3) if grid is not None else model)
            )
            for name, model in candidates.items()
            for grid in [param_source(name, rf_n_estimators, rf_max_depth, gb_learning_rate, gb_max_iter, logreg_C)]
        }

    scoring_spec = _cv_scoring_spec(task, n_classes)
    comparison: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, object] = {}
    primary_metric_folds: dict[str, np.ndarray] = {}

    for name, model in candidates.items():
        needs_sample_weight = sample_weight is not None and name == "Gradient Boosting"
        fit_kwargs = {"sample_weight": sample_weight} if needs_sample_weight else {}
        if use_cv:
            comparison[name], primary_metric_folds[name] = _cross_validate_metrics(
                model, X_full, y_full, cv_folds, scoring_spec, primary_metric
            )
        else:
            model.fit(X_train, y_train, **fit_kwargs)
            preds = model.predict(X_test)
            if task == "regression":
                comparison[name] = {
                    "MAE": float(mean_absolute_error(y_test, preds)),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
                    "R2": float(r2_score(y_test, preds)),
                }
            else:
                row = {
                    "Accuracy": float(accuracy_score(y_test, preds)),
                    "F1_weighted": float(f1_score(y_test, preds, average="weighted")),
                    "Balanced_Accuracy": float(balanced_accuracy_score(y_test, preds)),
                }
                if n_classes == 2:
                    proba = model.predict_proba(X_test)[:, 1]
                    row["ROC_AUC"] = float(roc_auc_score(y_test, proba))
                comparison[name] = row
        fitted_models[name] = model

    best_name = max(comparison, key=lambda n: comparison[n][primary_metric])
    scoring = "r2" if task == "regression" else "f1_weighted"

    result.model_comparison = comparison
    result.best_model_name = best_name
    result.metrics = comparison[best_name]
    best_model = fitted_models[best_name]

    if use_cv:
        # cross_validate fits its own internal clones; the object we hold
        # onto here has never been fit. Refit on the standard single split
        # so we have a concrete model to compute diagnostics/importance/tree
        # against -- the headline metrics above remain the CV means.
        best_fit_kwargs = (
            {"sample_weight": sample_weight} if (sample_weight is not None and best_name == "Gradient Boosting") else {}
        )
        best_model.fit(X_train, y_train, **best_fit_kwargs)

    result.hyperparameter_search_applied = do_grid_search
    result.broad_hyperparameter_search_applied = do_broad_search
    result.best_hyperparameters = _extract_fitted_hyperparameters(best_name, best_model)

    if use_cv and len(comparison) >= 2:
        runner_up_name = sorted(comparison, key=lambda n: comparison[n][primary_metric], reverse=True)[1]
        result.runner_up_model_name = runner_up_name
        best_folds = primary_metric_folds.get(best_name)
        runner_up_folds = primary_metric_folds.get(runner_up_name)
        if best_folds is not None and runner_up_folds is not None:
            try:
                stat, p_value = wilcoxon(best_folds, runner_up_folds)
                result.best_vs_runner_up_test = AssociationTest(
                    feature=f"{best_name} vs {runner_up_name}",
                    test_name="Wilcoxon signed-rank",
                    statistic=float(stat),
                    p_value=float(p_value),
                    significant=bool(p_value < 0.05),
                )
            except ValueError:
                pass  # e.g. identical per-fold scores -- the test is undefined, not a failure

    if task == "classification":
        best_preds = best_model.predict(X_test)
        result.confusion = confusion_matrix(y_test, best_preds).tolist()
        result.class_labels = class_labels

    importance = permutation_importance(
        best_model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, scoring=scoring
    )
    imp_series = pd.Series(importance.importances_mean, index=X_train.columns).sort_values(ascending=False)
    result.feature_importances = {k: float(v) for k, v in imp_series.head(15).items()}

    result.is_tree_based = best_name in TREE_BASED_MODELS
    if result.is_tree_based:
        # The winning model is an ensemble of many (deeper) trees averaged
        # together -- not something you can draw. This shallow single tree,
        # fit fresh on the same split, is not the production model; it is a
        # faithful illustration of the split logic that kind of ensemble is
        # built from.
        if task == "regression":
            illustrative = DecisionTreeRegressor(max_depth=ILLUSTRATIVE_TREE_MAX_DEPTH, random_state=RANDOM_STATE)
        else:
            illustrative = DecisionTreeClassifier(max_depth=ILLUSTRATIVE_TREE_MAX_DEPTH, random_state=RANDOM_STATE)
        illustrative.fit(X_train, y_train)
        result.illustrative_tree = illustrative
        result.illustrative_tree_features = list(X_train.columns)

    return result
