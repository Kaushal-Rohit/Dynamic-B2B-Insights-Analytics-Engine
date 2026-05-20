from __future__ import annotations

import json
import os
from typing import Iterable, Optional

import pandas as pd

from data_input import validate_pipeline_columns


def _clean_cell_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _normalize_text_columns(
    text_column=None,
    text_columns: Optional[Iterable[str]] = None,
) -> Optional[list[str]]:
    selected_text_columns = text_columns if text_columns is not None else text_column

    if selected_text_columns is None:
        return None
    if isinstance(selected_text_columns, str):
        return [selected_text_columns]
    return list(selected_text_columns)


def standardize_dataframe(
    df: pd.DataFrame,
    text_column=None,
    text_columns: Optional[Iterable[str]] = None,
    metadata_columns: Optional[Iterable[str]] = None,
    title_column: Optional[str] = None,
    save_filename: str = "standardized_data.csv",
    data_folder: str = "data",
) -> pd.DataFrame:
    """Convert raw ticket data into title/content/metadata rows."""
    selected_text_columns = _normalize_text_columns(text_column, text_columns)
    selected_text_columns, title_column, metadata_columns = validate_pipeline_columns(
        df,
        text_columns=selected_text_columns,
        title_column=title_column,
        metadata_columns=metadata_columns,
    )

    standardized_rows = []

    for _, row in df.iterrows():
        title = ""
        if title_column:
            title = str(_clean_cell_value(row[title_column])).strip()

        content_parts = []
        for column in selected_text_columns:
            value = _clean_cell_value(row[column])
            if value != "":
                content_parts.append(str(value).strip())

        metadata = {}
        for column in metadata_columns:
            value = _clean_cell_value(row[column])
            metadata[column] = value

        standardized_rows.append(
            {
                "title": title,
                "content": " ".join(content_parts).strip(),
                "metadata": json.dumps(metadata, default=str),
            }
        )

    standardized_df = pd.DataFrame(standardized_rows)

    os.makedirs(data_folder, exist_ok=True)
    save_path = os.path.join(data_folder, save_filename)
    standardized_df.to_csv(save_path, index=False)
    standardized_df.attrs["save_path"] = save_path

    print(f"[INFO] Cleaned data saved at: {save_path}")
    return standardized_df


def standardize_data(
    df: pd.DataFrame,
    text_column=None,
    text_columns: Optional[Iterable[str]] = None,
    metadata_columns: Optional[Iterable[str]] = None,
    title_column: Optional[str] = None,
    save_filename: str = "standardized_data.csv",
    data_folder: str = "data",
) -> str:
    """Standardize a dataframe and return the path to the saved CSV file."""
    standardized_df = standardize_dataframe(
        df=df,
        text_column=text_column,
        text_columns=text_columns,
        metadata_columns=metadata_columns,
        title_column=title_column,
        save_filename=save_filename,
        data_folder=data_folder,
    )
    return standardized_df.attrs["save_path"]


if __name__ == "__main__":
    from data_input import (
        DEFAULT_METADATA_COLUMNS,
        DEFAULT_RAW_CSV,
        DEFAULT_TEXT_COLUMNS,
        DEFAULT_TITLE_COLUMN,
        load_raw_data,
    )

    raw_df = load_raw_data(DEFAULT_RAW_CSV)
    clean_data = standardize_dataframe(
        df=raw_df,
        text_columns=DEFAULT_TEXT_COLUMNS,
        title_column=DEFAULT_TITLE_COLUMN,
        metadata_columns=DEFAULT_METADATA_COLUMNS,
    )
    print(clean_data.head())
