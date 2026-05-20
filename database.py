from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd


class VectorDatabaseManager:
    def __init__(
        self,
        persist_directory: str = "data/vector_store",
        collection_name: str = "b2b_insights",
        reset_collection: bool = False,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.reset_collection = reset_collection
        self.client = None
        self.collection = None

        self._initialize_store()

    def _initialize_store(self):
        """Create the vector-store folder and connect to ChromaDB."""
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is required for the vector database. Install it with: pip install chromadb"
            ) from exc

        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        if self.reset_collection:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"[INFO] Reset existing collection: {self.collection_name}")
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "B2B Market Insights Vector Store"},
        )
        print(f"[INFO] Initialized vector store collection: {self.collection_name}")

    @staticmethod
    def _clean_metadata_value(value: Any):
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    @classmethod
    def _parse_metadata(cls, metadata_value: Any) -> dict:
        if metadata_value is None:
            return {}
        if isinstance(metadata_value, str) and metadata_value == "":
            return {}
        try:
            if pd.isna(metadata_value):
                return {}
        except (TypeError, ValueError):
            pass

        if isinstance(metadata_value, dict):
            metadata = metadata_value
        else:
            metadata = json.loads(str(metadata_value))

        return {
            str(key): cls._clean_metadata_value(value)
            for key, value in metadata.items()
        }

    def upsert_data(self, csv_path: str, embeddings_path: str, batch_size: int = 1000):
        """Load standardized data and embeddings into the ChromaDB collection."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find standardized CSV file: {csv_path}")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Could not find embeddings file: {embeddings_path}")

        print("[INFO] Loading standardized data and embeddings")
        df = pd.read_csv(csv_path)
        embeddings_array = np.load(embeddings_path)

        if len(df) != len(embeddings_array):
            raise ValueError(
                f"Row count mismatch: {len(df)} CSV rows but {len(embeddings_array)} embeddings"
            )

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for index, row in df.iterrows():
            ids.append(f"ticket-{index:06d}")
            documents.append(str(row.get("content", "")))
            embeddings.append(embeddings_array[index].tolist())

            metadata = self._parse_metadata(row.get("metadata", ""))
            metadata["title"] = str(row.get("title", ""))
            metadatas.append(metadata)

        print(f"[INFO] Upserting {len(ids)} records into ChromaDB")
        for start_index in range(0, len(ids), batch_size):
            end_index = min(start_index + batch_size, len(ids))
            self.collection.upsert(
                ids=ids[start_index:end_index],
                embeddings=embeddings[start_index:end_index],
                documents=documents[start_index:end_index],
                metadatas=metadatas[start_index:end_index],
            )
            print(f"       -> Uploaded rows {start_index} to {end_index}")

        print(f"[INFO] Vector DB now holds {self.collection.count()} records")
        return self.collection


def setup_vector_db(
    csv_path: str,
    embeddings_path: str,
    persist_directory: str = "data/vector_store",
    collection_name: str = "b2b_insights",
    reset_collection: bool = True,
):
    """Create/populate the vector database and return the Chroma collection."""
    db_manager = VectorDatabaseManager(
        persist_directory=persist_directory,
        collection_name=collection_name,
        reset_collection=reset_collection,
    )
    return db_manager.upsert_data(csv_path=csv_path, embeddings_path=embeddings_path)


if __name__ == "__main__":
    setup_vector_db(
        csv_path=os.path.join("data", "standardized_data.csv"),
        embeddings_path=os.path.join("embeddings", "embeddings.npy"),
    )
