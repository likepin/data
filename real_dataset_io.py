from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def normalize_date_col(date_col: str | None) -> str | None:
    if date_col is None:
        return None
    value = str(date_col).strip()
    if not value or value.lower() in {"none", "null", "na"}:
        return None
    return value


def normalize_header_mode(header_mode: str | None) -> str:
    value = "infer" if header_mode is None else str(header_mode).strip().lower()
    if value in {"", "infer", "header"}:
        return "infer"
    if value in {"none", "noheader", "header_none"}:
        return "none"
    raise ValueError(f"Unsupported header mode: {header_mode}")


def read_real_dataframe(
    data_path: Path,
    *,
    date_col: str | None = "date",
    header_mode: str | None = "infer",
    sep: str = ",",
) -> pd.DataFrame:
    resolved_header_mode = normalize_header_mode(header_mode)
    resolved_date_col = normalize_date_col(date_col)
    header = 0 if resolved_header_mode == "infer" else None
    df = pd.read_csv(data_path, header=header, sep=sep)
    if resolved_header_mode == "none":
        df.columns = [f"var_{idx}" for idx in range(df.shape[1])]
    if resolved_date_col and resolved_date_col not in df.columns:
        raise ValueError(f"date column not found in {data_path}: {resolved_date_col}")
    return df


def resolve_value_columns(
    df: pd.DataFrame,
    *,
    date_col: str | None = "date",
    value_cols: list[str] | None = None,
) -> list[str]:
    resolved_date_col = normalize_date_col(date_col)
    if value_cols:
        missing = [col for col in value_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Requested value columns missing from data: {missing}")
        return [str(col) for col in value_cols]
    return [str(col) for col in df.columns if col != resolved_date_col]


def load_values_matrix(
    data_path: Path,
    *,
    date_col: str | None = "date",
    value_cols: list[str] | None = None,
    header_mode: str | None = "infer",
    sep: str = ",",
) -> tuple[np.ndarray, list[str]]:
    df = read_real_dataframe(
        data_path,
        date_col=date_col,
        header_mode=header_mode,
        sep=sep,
    )
    resolved_value_cols = resolve_value_columns(df, date_col=date_col, value_cols=value_cols)
    values = df[resolved_value_cols].to_numpy(dtype=np.float64)
    return values, resolved_value_cols


def count_data_rows(
    data_path: Path,
    *,
    header_mode: str | None = "infer",
) -> int:
    resolved_header_mode = normalize_header_mode(header_mode)
    with data_path.open("r", encoding="utf-8") as handle:
        line_count = sum(1 for line in handle if line.strip())
    if resolved_header_mode == "infer":
        return max(0, line_count - 1)
    return line_count
