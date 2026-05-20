from __future__ import annotations

import json
import os
from typing import Any, Optional


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama3:latest"


def _load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for search. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    return SentenceTransformer(model_name)


def _build_where_filter(source_filter=None, category_filter=None):
    conditions = []

    if source_filter:
        conditions.append({"source": source_filter})
    if category_filter:
        conditions.append({"category": category_filter})

    if len(conditions) == 1:
        return conditions[0]
    if len(conditions) > 1:
        return {"$and": conditions}
    return None


def filtered_vector_search(
    query_text: str,
    db_collection,
    source_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    top_k: int = 50,
    embedding_model=None,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> dict:
    """Run a vector search against a ChromaDB collection."""
    if not query_text or not query_text.strip():
        raise ValueError("query_text cannot be empty")

    model = embedding_model or _load_embedding_model(embedding_model_name)
    query_embedding = model.encode(query_text).tolist()
    where_filter = _build_where_filter(source_filter, category_filter)

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances", "embeddings"],
    }
    if where_filter:
        query_args["where"] = where_filter

    return db_collection.query(**query_args)


def _empty_cluster_frame():
    import pandas as pd

    return pd.DataFrame(columns=["Document", "Cluster_ID", "X_coord", "Y_coord"])


def _single_cluster_frame(documents):
    import pandas as pd

    return pd.DataFrame(
        {
            "Document": documents,
            "Cluster_ID": [0] * len(documents),
            "X_coord": [0.0] * len(documents),
            "Y_coord": [0.0] * len(documents),
        }
    )


def cluster_retrieved_tickets(embeddings, documents, min_cluster_size: int = 5):
    """Reduce dimensions and cluster retrieved tickets."""
    if not documents:
        return _empty_cluster_frame()

    if embeddings is None or len(embeddings) == 0 or len(documents) < 3:
        print("[WARN] Not enough embeddings for clustering; grouping results into one cluster.")
        return _single_cluster_frame(documents)

    try:
        import hdbscan
        import numpy as np
        import pandas as pd
        import umap
    except ImportError:
        print("[WARN] umap-learn or hdbscan is not installed; grouping results into one cluster.")
        return _single_cluster_frame(documents)

    embeddings_array = np.array(embeddings)
    if embeddings_array.ndim != 2 or embeddings_array.shape[0] != len(documents):
        print("[WARN] Embedding shape does not match documents; grouping results into one cluster.")
        return _single_cluster_frame(documents)

    safe_n_neighbors = min(15, len(embeddings_array) - 1)
    safe_n_neighbors = max(2, safe_n_neighbors)
    effective_min_cluster_size = max(2, min(min_cluster_size, len(documents)))

    print(f"[INFO] Reducing dimensions with UMAP (n_neighbors={safe_n_neighbors})")
    reducer = umap.UMAP(
        n_neighbors=safe_n_neighbors,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    reduced_embeddings = reducer.fit_transform(embeddings_array)

    print(f"[INFO] Clustering with HDBSCAN (min_cluster_size={effective_min_cluster_size})")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=effective_min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    cluster_labels = clusterer.fit_predict(reduced_embeddings)

    results_df = pd.DataFrame(
        {
            "Document": documents,
            "Cluster_ID": cluster_labels,
            "X_coord": reduced_embeddings[:, 0],
            "Y_coord": reduced_embeddings[:, 1],
        }
    )

    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"[INFO] Discovered {num_clusters} topic cluster(s)")
    return results_df


def prepare_clusters_for_llm(clustered_df, save_path: str = "data/llm_ready_clusters.json") -> dict:
    """Format clustered tickets as JSON for the LLM step."""
    print("[INFO] Formatting clusters for LLM ingestion")

    clean_df = clustered_df[clustered_df["Cluster_ID"] != -1]
    grouped_data = clean_df.groupby("Cluster_ID")["Document"].apply(list).to_dict()

    llm_payload = {}
    for cluster_id, documents in grouped_data.items():
        cluster_name = f"Cluster_{cluster_id}"
        llm_payload[cluster_name] = {
            "ticket_count": len(documents),
            "sample_tickets": documents,
        }

    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as output_file:
        json.dump(llm_payload, output_file, indent=4)

    print(f"[INFO] Saved {len(llm_payload)} cluster(s) to {save_path}")
    return llm_payload


class B2BInsightGenerator:
    def __init__(
        self,
        input_file: str = "data/llm_ready_clusters.json",
        output_file: str = "data/final_b2b_insights.json",
        model_name: str = DEFAULT_LLM_MODEL,
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.model_name = model_name
        self.clusters_data = {}

    def load_data(self) -> bool:
        if not os.path.exists(self.input_file):
            print(f"[ERROR] Could not find {self.input_file}. Run clustering first.")
            return False

        print(f"[INFO] Loading clustered data from {self.input_file}")
        with open(self.input_file, "r", encoding="utf-8") as input_file:
            self.clusters_data = json.load(input_file)

        print(f"[INFO] Loaded {len(self.clusters_data)} cluster(s) for analysis")
        return True

    def generate_insights(self) -> dict:
        if not self.clusters_data:
            print("[WARN] No cluster data available for insight generation.")
            return {}

        try:
            import ollama
        except ImportError:
            print("[WARN] ollama is not installed; skipping LLM insight generation.")
            return {}

        print(f"[INFO] Generating insights with Ollama model: {self.model_name}")
        final_insights = {}

        system_prompt = """
You are an expert B2B product analyst. Read clustered customer support tickets.

Respond strictly in this format:
Title: [A short, 3-5 word name for the issue]
Summary: [A concise 1-sentence summary of what is happening]
Actionable Advice: [One step the product engineering team should take to fix it]
"""

        for cluster_name, data in self.clusters_data.items():
            ticket_count = data["ticket_count"]
            tickets_to_analyze = data["sample_tickets"][:10]

            print(f"\n[INFO] Analyzing {cluster_name} ({ticket_count} total tickets)")
            user_prompt = f"Here are the tickets to analyze:\n{json.dumps(tickets_to_analyze, indent=2)}"

            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            except Exception as exc:
                print(f"[ERROR] Failed to communicate with Ollama for {cluster_name}: {exc}")
                continue

            llm_output = response["message"]["content"]
            print(f"\n--- {cluster_name} Insight ---")
            print(llm_output)
            print("-" * 40)

            final_insights[cluster_name] = {
                "ticket_count": ticket_count,
                "llm_analysis": llm_output,
            }

        self._save_results(final_insights)
        return final_insights

    def _save_results(self, final_insights: dict):
        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as output_file:
            json.dump(final_insights, output_file, indent=4)

        print(f"[INFO] Insights saved to {self.output_file}")


def _first_result_list(results: dict, key: str) -> list:
    value = results.get(key)
    if value is None or len(value) == 0:
        return []
    first_value = value[0]
    if first_value is None:
        return []
    return first_value


def print_search_results(results: dict, limit: int = 5):
    documents = _first_result_list(results, "documents")
    metadatas = _first_result_list(results, "metadatas")
    distances = _first_result_list(results, "distances")

    print("\n===== SEARCH RESULTS =====\n")
    for index, document in enumerate(documents[:limit]):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        distance_text = f"{distance:.4f}" if isinstance(distance, (int, float)) else "n/a"

        print(f"Result {index + 1}")
        print("-" * 50)
        print("Document:", document)
        print("Metadata:", metadata)
        print(f"Distance Score: {distance_text}\n")


def query_and_analyze(
    query_text: str,
    db_collection,
    source_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    top_k: int = 50,
    min_cluster_size: int = 5,
    clusters_path: str = "data/llm_ready_clusters.json",
    insights_path: str = "data/final_b2b_insights.json",
    embedding_model=None,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    llm_model_name: str = DEFAULT_LLM_MODEL,
    run_llm: bool = True,
) -> dict[str, Any]:
    """Search, cluster, and optionally summarize matching tickets."""
    search_results = filtered_vector_search(
        query_text=query_text,
        db_collection=db_collection,
        source_filter=source_filter,
        category_filter=category_filter,
        top_k=top_k,
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
    )

    print_search_results(search_results)

    documents = _first_result_list(search_results, "documents")
    embeddings = _first_result_list(search_results, "embeddings")
    if not documents:
        print("[INFO] No matching documents found.")
        return {"search_results": search_results, "clusters": {}, "insights": {}}

    clustered_data = cluster_retrieved_tickets(
        embeddings=embeddings,
        documents=documents,
        min_cluster_size=min_cluster_size,
    )
    llm_payload = prepare_clusters_for_llm(clustered_data, save_path=clusters_path)

    insights = {}
    if run_llm and llm_payload:
        insight_engine = B2BInsightGenerator(
            input_file=clusters_path,
            output_file=insights_path,
            model_name=llm_model_name,
        )
        if insight_engine.load_data():
            insights = insight_engine.generate_insights()

    return {
        "search_results": search_results,
        "clustered_data": clustered_data,
        "clusters": llm_payload,
        "insights": insights,
    }


if __name__ == "__main__":
    import chromadb

    client = chromadb.PersistentClient(path="data/vector_store")
    collection = client.get_collection(name="b2b_insights")
    query_and_analyze(query_text="sound", db_collection=collection)
