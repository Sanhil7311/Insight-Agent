from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

_NULL_DROP_THRESHOLD: float = 0.50
_IQR_MULTIPLIER: float = 1.5
_DATE_MATCH_THRESHOLD: float = 0.80
_DATE_PATTERN: re.Pattern[str] = re.compile(
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"
)
_DATE_SAMPLE_SIZE: int = 20
_TOP_CATEGORY_K: int = 5
_CORR_THRESHOLD: float = 0.30
_MAX_CHARTS: int = 5
_AUTOML_TEST_SIZE: float = 0.20
_AUTOML_RANDOM_STATE: int = 42
_AUTOML_N_ESTIMATORS: int = 100
_AUTOML_TOP_FEATURES: int = 10
_AUTOML_MAX_CARDINALITY: int = 50


def _safe_round(value: float, decimals: int = 3) -> float:
    return round(float(value), decimals)


def _to_python_scalar(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_python_scalar(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python_scalar(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


@lru_cache(maxsize=8)
def load_dataset(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path!r}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _load_csv(file_path)

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        logger.debug("Loading Excel file: %s", file_path)
        return pd.read_excel(file_path, engine="openpyxl")

    raise ValueError(
        f"Unsupported file extension {suffix!r}. "
        "Expected one of: .csv, .xlsx, .xls, .xlsm"
    )


def _load_csv(file_path: str) -> pd.DataFrame:
    logger.debug("Loading CSV file: %s", file_path)
    try:
        return pd.read_csv(file_path, engine="c", low_memory=False)
    except UnicodeDecodeError:
        logger.warning(
            "UTF-8 decode failed for %r — retrying with latin1.", file_path
        )
        return pd.read_csv(file_path, encoding="latin1", engine="c", low_memory=False)


def run_cleaning(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    report: dict[str, Any] = {
        "issues": [],
        "actions": [],
        "summary": {
            "rows_removed": 0,
            "columns_removed": 0,
            "nulls_filled": 0,
            "outliers_clipped": 0,
        },
    }

    df, report = _remove_duplicates(df, report)
    df, report = _handle_nulls(df, report)
    df, report = _clip_outliers(df, report)
    df, report = _parse_datetimes(df, report)

    return df, report


def _remove_duplicates(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, dict]:
    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        df = df.drop_duplicates()
        msg = f"Dropped {n_dup} exact duplicate row{'s' if n_dup > 1 else ''}."
        report["issues"].append(f"Found {n_dup} duplicate rows.")
        report["actions"].append(msg)
        report["summary"]["rows_removed"] += n_dup
        logger.info(msg)
    return df, report


def _handle_nulls(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, dict]:
    cols_to_drop: list[str] = []
    for col in df.columns:
        n_null = int(df[col].isnull().sum())
        if n_null == 0:
            continue
        pct_null = n_null / len(df)

        if pct_null > _NULL_DROP_THRESHOLD:
            cols_to_drop.append(col)
            report["issues"].append(
                f"Column '{col}' has {pct_null:.0%} missing values — will be dropped."
            )
            report["actions"].append(
                f"Dropped column '{col}' ({pct_null:.0%} null)."
            )
            report["summary"]["columns_removed"] += 1
        else:
            dtype_kind = df[col].dtype.kind
            if dtype_kind in "iufcb":
                fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
                msg = (
                    f"Filled {n_null} null(s) in numeric column '{col}' "
                    f"with median ({_safe_round(fill_value, 4)})."
                )
            else:
                mode_series = df[col].mode()
                fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                df[col] = df[col].fillna(fill_value)
                msg = (
                    f"Filled {n_null} null(s) in categorical column '{col}' "
                    f"with mode ({fill_value!r})."
                )
            report["issues"].append(
                f"Column '{col}' has {n_null} missing value(s) ({pct_null:.1%})."
            )
            report["actions"].append(msg)
            report["summary"]["nulls_filled"] += n_null
            logger.debug(msg)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df, report


def _clip_outliers(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, dict]:
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1

        if iqr == 0.0:
            continue

        lower_fence = q1 - _IQR_MULTIPLIER * iqr
        upper_fence = q3 + _IQR_MULTIPLIER * iqr

        mask_out = (df[col] < lower_fence) | (df[col] > upper_fence)
        n_out = int(mask_out.sum())

        if n_out > 0:
            df[col] = df[col].clip(lower=lower_fence, upper=upper_fence)
            msg = (
                f"Clipped {n_out} outlier(s) in '{col}' "
                f"to fence [{_safe_round(lower_fence, 3)}, {_safe_round(upper_fence, 3)}]."
            )
            report["issues"].append(
                f"Column '{col}' has {n_out} outlier(s) beyond IQR fence."
            )
            report["actions"].append(msg)
            report["summary"]["outliers_clipped"] += n_out
            logger.debug(msg)

    return df, report


def _parse_datetimes(df: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, dict]:
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(_DATE_SAMPLE_SIZE).astype(str)

        if sample.empty:
            continue

        match_rate = sample.str.match(_DATE_PATTERN).mean()

        if match_rate >= _DATE_MATCH_THRESHOLD:
            converted = pd.to_datetime(df[col], errors="coerce")
            n_failed = int(converted.isnull().sum()) - int(df[col].isnull().sum())

            if n_failed / max(len(df), 1) < 0.20:
                df[col] = converted
                msg = (
                    f"Parsed column '{col}' as datetime "
                    f"({match_rate:.0%} pattern match rate)."
                )
                report["actions"].append(msg)
                logger.debug(msg)

    return df, report


def compute_analytics(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include="number")
    cat_df = df.select_dtypes(include="object")

    analytics: dict[str, Any] = {
        "shape": list(df.shape),
        "numeric_summary": _compute_numeric_summary(numeric_df),
        "correlations": _compute_correlations(numeric_df),
        "categories": _compute_categories(cat_df),
        "null_counts": _compute_null_counts(df),
        "dtype_map": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    return _to_python_scalar(analytics)


def _compute_numeric_summary(numeric_df: pd.DataFrame) -> dict:
    summary: dict[str, dict] = {}

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        summary[col] = {
            "mean": _safe_round(series.mean()),
            "std": _safe_round(series.std()),
            "min": _safe_round(series.min()),
            "max": _safe_round(series.max()),
            "median": _safe_round(series.median()),
            "skew": _safe_round(series.skew()),
            "kurtosis": _safe_round(series.kurt()),
            "q25": _safe_round(series.quantile(0.25)),
            "q75": _safe_round(series.quantile(0.75)),
            "non_null_count": int(series.count()),
        }

    return summary


def _compute_correlations(numeric_df: pd.DataFrame) -> dict:
    if numeric_df.shape[1] < 2:
        return {}

    corr_matrix = numeric_df.corr(method="pearson")
    pairs: dict[str, float] = {}
    cols = corr_matrix.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            r = corr_matrix.iloc[i, j]
            if pd.isna(r):
                continue
            if abs(r) > _CORR_THRESHOLD:
                pairs[f"{a}×{b}"] = _safe_round(r)

    return dict(sorted(pairs.items(), key=lambda kv: abs(kv[1]), reverse=True))


def _compute_categories(cat_df: pd.DataFrame) -> dict:
    categories: dict[str, dict] = {}

    for col in cat_df.columns:
        top_k = (
            cat_df[col]
            .value_counts(dropna=True)
            .head(_TOP_CATEGORY_K)
            .to_dict()
        )
        categories[col] = {str(k): int(v) for k, v in top_k.items()}

    return categories


def _compute_null_counts(df: pd.DataFrame) -> dict:
    return {col: int(count) for col, count in df.isnull().sum().items()}


def _prepare_automl_features(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series] | None:
    if target_column not in df.columns:
        logger.warning("Target column %r not found in DataFrame.", target_column)
        return None

    df_work = df.copy()
    y_raw = df_work[target_column]
    X_raw = df_work.drop(columns=[target_column])

    datetime_cols = X_raw.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    X_raw = X_raw.drop(columns=datetime_cols)

    object_cols = X_raw.select_dtypes(include="object").columns.tolist()
    for col in object_cols:
        cardinality = X_raw[col].nunique()
        if cardinality > _AUTOML_MAX_CARDINALITY:
            X_raw = X_raw.drop(columns=[col])
        else:
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))

    X_raw = X_raw.select_dtypes(include="number")

    if X_raw.empty or X_raw.shape[1] == 0:
        logger.warning("No usable feature columns remain after preprocessing.")
        return None

    X_raw = X_raw.fillna(X_raw.median())

    return X_raw, y_raw


def run_automl(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> dict[str, Any]:
    """Train a baseline RandomForest model and return metrics and feature importances.

    Parameters
    ----------
    df:
        Cleaned DataFrame.
    target_column:
        Name of the column to predict.
    task_type:
        Either ``"classification"`` or ``"regression"``.

    Returns
    -------
    dict
        Keys: ``task_type``, ``target_column``, ``model``, ``train_samples``,
        ``test_samples``, ``metrics``, ``feature_importances``, and optionally
        ``"error"`` if training failed.
    """
    result: dict[str, Any] = {
        "task_type": task_type,
        "target_column": target_column,
        "model": "RandomForest",
        "train_samples": 0,
        "test_samples": 0,
        "metrics": {},
        "feature_importances": {},
        "skipped": False,
    }

    try:
        prepared = _prepare_automl_features(df, target_column)
        if prepared is None:
            result["error"] = (
                "Could not prepare features. Check that the dataset has usable "
                "numeric or low-cardinality categorical columns."
            )
            return result

        X, y = prepared

        if len(X) < 20:
            result["error"] = (
                f"Dataset has only {len(X)} rows after preprocessing — "
                "too few for a reliable train/test split."
            )
            return result

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=_AUTOML_TEST_SIZE,
            random_state=_AUTOML_RANDOM_STATE,
        )

        result["train_samples"] = int(len(X_train))
        result["test_samples"] = int(len(X_test))

        if task_type == "classification":
            y_train_enc = LabelEncoder().fit(y_train)
            y_train_t = y_train_enc.transform(y_train)
            y_test_t = y_train_enc.transform(
                pd.Series(y_test).map(
                    lambda v: v if v in y_train_enc.classes_ else y_train_enc.classes_[0]
                )
            )
            model = RandomForestClassifier(
                n_estimators=_AUTOML_N_ESTIMATORS,
                random_state=_AUTOML_RANDOM_STATE,
                n_jobs=-1,
            )
            model.fit(X_train, y_train_t)
            y_pred = model.predict(X_test)

            avg = "binary" if len(y_train_enc.classes_) == 2 else "weighted"
            result["metrics"] = {
                "accuracy": _safe_round(accuracy_score(y_test_t, y_pred), 4),
                "f1_score": _safe_round(f1_score(y_test_t, y_pred, average=avg, zero_division=0), 4),
            }

        else:
            if not pd.api.types.is_numeric_dtype(y):
                result["error"] = (
                    f"Target column '{target_column}' is not numeric. "
                    "Choose a numeric column for Regression or switch to Classification."
                )
                return result

            model = RandomForestRegressor(
                n_estimators=_AUTOML_N_ESTIMATORS,
                random_state=_AUTOML_RANDOM_STATE,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            result["metrics"] = {
                "rmse": _safe_round(float(np.sqrt(mse)), 4),
                "r2_score": _safe_round(float(r2_score(y_test, y_pred)), 4),
            }

        importances = model.feature_importances_
        feature_names = X.columns.tolist()
        fi_pairs = sorted(
            zip(feature_names, importances),
            key=lambda p: p[1],
            reverse=True,
        )[:_AUTOML_TOP_FEATURES]
        result["feature_importances"] = {
            name: _safe_round(float(imp), 4) for name, imp in fi_pairs
        }

        logger.info(
            "AutoML complete | task=%s | target=%r | metrics=%s",
            task_type, target_column, result["metrics"],
        )

    except Exception as exc:
        logger.exception("AutoML failed: %s", exc)
        result["error"] = f"{type(exc).__name__}: {exc}"

    return _to_python_scalar(result)


def recommend_charts(df: pd.DataFrame) -> list[dict]:
    num_cols: list[str] = df.select_dtypes(include="number").columns.tolist()
    cat_cols: list[str] = df.select_dtypes(include="object").columns.tolist()
    dt_cols: list[str] = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    recommendations: list[dict] = []

    if num_cols:
        recommendations.append(
            {
                "chart_type": "histogram",
                "x_axis": num_cols[0],
                "y_axis": None,
                "title": f"Distribution of {num_cols[0]}",
            }
        )

    if cat_cols and num_cols:
        recommendations.append(
            {
                "chart_type": "bar",
                "x_axis": cat_cols[0],
                "y_axis": num_cols[0],
                "title": f"{num_cols[0]} by {cat_cols[0]}",
            }
        )

    if len(num_cols) >= 3:
        recommendations.append(
            {
                "chart_type": "heatmap",
                "x_axis": None,
                "y_axis": None,
                "title": "Numeric Correlation Heatmap",
            }
        )

    if len(num_cols) >= 2:
        best_pair = _find_strongest_correlation_pair(df[num_cols])
        if best_pair:
            col_a, col_b = best_pair
            recommendations.append(
                {
                    "chart_type": "scatter",
                    "x_axis": col_a,
                    "y_axis": col_b,
                    "title": f"{col_a} vs {col_b}",
                }
            )

    if cat_cols and num_cols:
        y_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]
        recommendations.append(
            {
                "chart_type": "box",
                "x_axis": cat_cols[0],
                "y_axis": y_col,
                "title": f"{y_col} Distribution by {cat_cols[0]}",
            }
        )

    if dt_cols and num_cols and len(recommendations) < _MAX_CHARTS:
        recommendations.append(
            {
                "chart_type": "line",
                "x_axis": dt_cols[0],
                "y_axis": num_cols[0],
                "title": f"{num_cols[0]} Over Time ({dt_cols[0]})",
            }
        )

    return recommendations[:_MAX_CHARTS]


def _find_strongest_correlation_pair(
    numeric_df: pd.DataFrame,
) -> tuple[str, str] | None:
    if numeric_df.shape[1] < 2:
        return None

    corr_arr = numeric_df.corr(method="pearson").abs().values.copy()
    np.fill_diagonal(corr_arr, 0.0)
    corr = pd.DataFrame(corr_arr, index=numeric_df.columns, columns=numeric_df.columns)

    stacked = corr.stack()
    if stacked.empty or stacked.max() == 0.0:
        return (numeric_df.columns[0], numeric_df.columns[1])

    idx = stacked.idxmax()
    return (str(idx[0]), str(idx[1]))