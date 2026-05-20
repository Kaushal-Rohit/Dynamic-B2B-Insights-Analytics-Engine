from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd


EMBEDDING_FOLDER = "embeddings"
EMBEDDING_FILE = "embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"
TEXT_COLUMNS_TO_EMBED = ["title", "content"]


def _load_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required to generate embeddings. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    print(f"[INFO] Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def generate_embeddings(
    csv_path: str,
    embeddings_path: str | None = None,
    model_name: str = MODEL_NAME,
    text_columns: Iterable[str] = TEXT_COLUMNS_TO_EMBED,
    batch_size: int = 32,
) -> str:
    """Generate and save vector embeddings for a standardized CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find standardized CSV file: {csv_path}")

    df = pd.read_csv(csv_path)
    selected_text_columns = list(text_columns)
    missing_columns = [column for column in selected_text_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing column(s) in standardized data: {', '.join(missing_columns)}")
    if df.empty:
        raise ValueError("Cannot generate embeddings from an empty CSV file.")

    combined_text = (
        df[selected_text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
        .tolist()
    )

    output_path = embeddings_path or os.path.join(EMBEDDING_FOLDER, EMBEDDING_FILE)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    model = _load_embedding_model(model_name)
    print(f"[INFO] Generating embeddings for {len(combined_text)} rows")
    embeddings = model.encode(
        combined_text,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    np.save(output_path, embeddings)
    print(f"[INFO] Embeddings saved at: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_embeddings(os.path.join("data", "standardized_data.csv"))
