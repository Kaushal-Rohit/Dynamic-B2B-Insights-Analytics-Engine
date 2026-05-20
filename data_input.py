from __future__ import annotations

import os
from typing import Iterable, Optional

import pandas as pd


DEFAULT_RAW_CSV = "data.csv"
DEFAULT_TEXT_COLUMNS = ["Ticket Description", "Ticket Subject", "Ticket Type"]
DEFAULT_TITLE_COLUMN = "Product Purchased"
DEFAULT_METADATA_COLUMNS = [
    "Customer Age",
    "Customer Gender",
    "Date of Purchase",
    "Ticket Channel",
    "Customer Satisfaction Rating",
]


def normalize_csv_path(csv_path: Optional[str] = None) -> str:
    """Return a clean CSV path, falling back to the project sample file."""
    cleaned_path = (csv_path or DEFAULT_RAW_CSV).strip().strip('"').strip("'")
    return cleaned_path or DEFAULT_RAW_CSV


def load_raw_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load the raw CSV input for the pipeline."""
    resolved_path = normalize_csv_path(csv_path)

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Could not find CSV file: {resolved_path}")

    df = pd.read_csv(resolved_path)
    print(f"[INFO] Loaded raw data from {resolved_path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Return column names that are not present in the dataframe."""
    return [column for column in columns if column and column not in df.columns]


def validate_pipeline_columns(
    df: pd.DataFrame,
    text_columns: Optional[Iterable[str]] = None,
    title_column: Optional[str] = None,
    metadata_columns: Optional[Iterable[str]] = None,
) -> tuple[list[str], Optional[str], list[str]]:
    """Validate and normalize the column names used by the pipeline."""
    selected_text_columns = list(text_columns or DEFAULT_TEXT_COLUMNS)
    selected_title_column = title_column if title_column is not None else DEFAULT_TITLE_COLUMN
    selected_metadata_columns = list(
        DEFAULT_METADATA_COLUMNS if metadata_columns is None else metadata_columns
    )

    required_columns = list(selected_text_columns)
    if selected_title_column:
        required_columns.append(selected_title_column)

    missing_required = missing_columns(df, required_columns)
    if missing_required:
        available = ", ".join(df.columns)
        missing = ", ".join(missing_required)
        raise ValueError(f"Missing required column(s): {missing}. Available columns: {available}")

    missing_metadata = missing_columns(df, selected_metadata_columns)
    if missing_metadata:
        print(f"[WARN] Skipping missing metadata column(s): {', '.join(missing_metadata)}")
        selected_metadata_columns = [
            column for column in selected_metadata_columns if column in df.columns
        ]

    return selected_text_columns, selected_title_column, selected_metadata_columns
