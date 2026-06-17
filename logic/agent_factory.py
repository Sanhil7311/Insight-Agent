from __future__ import annotations

import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from crewai import LLM

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
_MODEL_ID: str = "ollama/insight-agent"
_TEMPERATURE: float = 0.2
_CORR_THRESHOLD: float = 0.30
_TOP_K_CATEGORIES: int = 5

_llm_lock: threading.Lock = threading.Lock()
_llm_instance: LLM | None = None


def get_local_llm() -> LLM:
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    with _llm_lock:
        if _llm_instance is not None:
            return _llm_instance

        logger.info(
            "Initialising singleton CrewAI LLM | model=%s | base_url=%s",
            _MODEL_ID,
            _OLLAMA_BASE_URL,
        )

        _llm_instance = LLM(
            model=_MODEL_ID,
            base_url=_OLLAMA_BASE_URL,
            temperature=_TEMPERATURE,
            max_tokens=1024,
            top_p=0.9,
            frequency_penalty=0.1,
            timeout=180,
        )

        logger.info("Singleton LLM ready.")
        return _llm_instance


def reset_llm_singleton() -> None:
    global _llm_instance
    with _llm_lock:
        _llm_instance = None
        logger.info("LLM singleton has been reset.")


def _safe_float(value: Any, decimals: int = 3) -> float:
    return round(float(value), decimals)


def _to_python(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@lru_cache(maxsize=8)
def _load_raw_dataframe(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path!r}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(file_path, engine="c", low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding="latin1", engine="c", low_memory=False)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(file_path, engine="openpyxl")
    raise ValueError(f"Unsupported file extension {suffix!r}.")


def _build_shape_section(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    return f"Shape: {rows} rows × {cols} columns\nColumns: {', '.join(df.columns.tolist())}\n"


def _build_dtype_section(df: pd.DataFrame) -> str:
    lines = ["Data Types:"]
    for col, dtype in df.dtypes.items():
        lines.append(f"  {col}: {dtype}")
    return "\n".join(lines) + "\n"


def _build_numeric_summary(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return "Numeric Summary: none\n"
    parts = ["Numeric Summary (mean | std | min | max | skew):"]
    for col in numeric.columns:
        s = numeric[col].dropna()
        if s.empty:
            continue
        parts.append(
            f"  {col}: mean={_safe_float(s.mean())} | std={_safe_float(s.std())} "
            f"| min={_safe_float(s.min())} | max={_safe_float(s.max())} "
            f"| skew={_safe_float(s.skew())}"
        )
    return "\n".join(parts) + "\n"


def _build_missing_profile(df: pd.DataFrame) -> str:
    null_counts = df.isnull().sum()
    missing = null_counts[null_counts > 0]
    if missing.empty:
        return "Missing Values: none\n"
    parts = ["Missing Values:"]
    for col, count in missing.items():
        pct = count / len(df) * 100
        parts.append(f"  {col}: {count} ({pct:.1f}%)")
    return "\n".join(parts) + "\n"


def _build_correlation_section(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return "Correlations: insufficient numeric columns\n"
    corr = numeric.corr(method="pearson")
    cols = corr.columns.tolist()
    pairs: list[str] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.isna(r) or abs(r) <= _CORR_THRESHOLD:
                continue
            pairs.append(f"  {cols[i]} × {cols[j]}: r={_safe_float(r)}")
    if not pairs:
        return f"Correlations: no pairs with |r| > {_CORR_THRESHOLD}\n"
    pairs.sort(key=lambda s: abs(float(s.split("r=")[1])), reverse=True)
    return "Correlations (|r| > 0.30):\n" + "\n".join(pairs) + "\n"


def _build_category_section(df: pd.DataFrame) -> str:
    cat = df.select_dtypes(include="object")
    if cat.empty:
        return "Categorical Columns: none\n"
    parts = [f"Categorical Columns (top {_TOP_K_CATEGORIES} values each):"]
    for col in cat.columns:
        top = cat[col].value_counts(dropna=True).head(_TOP_K_CATEGORIES)
        entries = ", ".join(f"{k!r}:{v}" for k, v in top.items())
        parts.append(f"  {col}: {entries}")
    return "\n".join(parts) + "\n"


def _get_data_context(file_path: str) -> str:
    df = _load_raw_dataframe(file_path)
    sections: list[str] = [
        f"=== Data Context: {Path(file_path).name} ===\n",
        _build_shape_section(df),
        _build_dtype_section(df),
        _build_numeric_summary(df),
        _build_missing_profile(df),
        _build_correlation_section(df),
        _build_category_section(df),
        "=== End of Data Context ===",
    ]
    context = "\n".join(sections)
    logger.debug("_get_data_context produced %d chars for %r", len(context), file_path)
    return context


def build_analytics_bundle(file_path: str) -> dict[str, Any]:
    df = _load_raw_dataframe(file_path)
    numeric = df.select_dtypes(include="number")
    cat = df.select_dtypes(include="object")

    numeric_summary: dict[str, dict] = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if s.empty:
            continue
        numeric_summary[col] = {
            "mean": _safe_float(s.mean()),
            "std": _safe_float(s.std()),
            "min": _safe_float(s.min()),
            "max": _safe_float(s.max()),
            "median": _safe_float(s.median()),
            "skew": _safe_float(s.skew()),
            "non_null_count": int(s.count()),
        }

    correlations: dict[str, float] = {}
    if numeric.shape[1] >= 2:
        corr = numeric.corr(method="pearson")
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iloc[i, j]
                if pd.isna(r) or abs(r) <= _CORR_THRESHOLD:
                    continue
                correlations[f"{cols[i]}×{cols[j]}"] = _safe_float(r)

    categories: dict[str, dict] = {
        col: {
            str(k): int(v)
            for k, v in cat[col].value_counts(dropna=True).head(_TOP_K_CATEGORIES).items()
        }
        for col in cat.columns
    }

    return _to_python({
        "shape": list(df.shape),
        "numeric_summary": numeric_summary,
        "correlations": correlations,
        "categories": categories,
        "null_counts": {col: int(n) for col, n in df.isnull().sum().items()},
        "dtype_map": {col: str(dt) for col, dt in df.dtypes.items()},
    })