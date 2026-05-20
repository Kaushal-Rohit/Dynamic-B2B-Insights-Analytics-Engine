# Dynamic B2B Insights Analytics Engine

An end-to-end analytics platform for turning unstructured B2B customer signals (tickets, complaints, product feedback, and support text) into actionable business insights. The project standardizes raw CSV input, generates semantic embeddings, indexes records in a persistent vector database, retrieves query-relevant records, clusters them into market themes, and generates LLM-powered recommendations.

## Project Description

This repository implements a practical **NLP + vector search + unsupervised clustering + LLM synthesis** workflow for market intelligence:

- **Schema-flexible ingestion** from raw CSVs
- **Standardization** into a consistent title/content/metadata structure
- **Sentence-transformer embeddings** for semantic similarity
- **ChromaDB vector storage** for scalable retrieval
- **UMAP + HDBSCAN clustering** to discover natural issue/theme groups
- **Ollama-based LLM analysis** to generate concise, actionable insight outputs

---

## Architecture and Pipeline Overview

### 1) Data Ingestion
- `main.py` starts an interactive pipeline builder.
- Users provide a CSV and map title/text/metadata columns.
- `data_input.py` validates required columns and handles defaults.

### 2) Data Standardization
- `cleaner.py` converts each row into:
  - `title`
  - `content` (merged text fields)
  - `metadata` (JSON string)
- Output is saved as a standardized CSV.

### 3) Embedding Generation
- `embedder.py` loads `all-MiniLM-L6-v2` (default).
- Concatenated title/content text is encoded into vectors.
- Vectors are saved as `.npy`.

### 4) Vector Database Indexing
- `database.py` initializes a persistent Chroma collection.
- Standardized rows + embeddings are batch-upserted.
- IDs, documents, and parsed metadata are stored for retrieval.

### 5) Query-Time Retrieval and Clustering
- `analyzer.py` converts user queries into embeddings.
- Top-k relevant records are fetched from ChromaDB.
- Retrieved vectors are reduced (UMAP) and clustered (HDBSCAN).
- Cluster payload is written to JSON for downstream analysis.

### 6) LLM-Powered Business Insights
- `B2BInsightGenerator` in `analyzer.py` uses Ollama.
- Each cluster is summarized into:
  - issue title
  - one-line summary
  - actionable product advice
- Final insights are saved as JSON for reporting and product review.

---

## Why UMAP + HDBSCAN (instead of KMeans or similar methods)?

### Why UMAP for dimensionality reduction
- Embedding vectors are high-dimensional and noisy; UMAP preserves local neighborhood structure better than linear reductions in many semantic-text settings.
- UMAP creates a compact manifold representation that improves density-based separation for downstream clustering.
- It is fast enough for interactive, query-time workflows.

### Why HDBSCAN for clustering
- Real support/feedback data has **uneven cluster sizes**, **non-spherical structure**, and **noise/outliers**.
- HDBSCAN handles variable-density clusters and labels ambiguous points as noise (`-1`) rather than forcing a wrong cluster assignment.
- It avoids hard-coding a fixed number of clusters.

### Why not KMeans as the default
- KMeans assumes roughly spherical clusters and similar variance, which rarely matches real-world customer text themes.
- KMeans requires preselecting `k`, which is unstable when issue volume and mix change by query.
- KMeans assigns every point to some cluster, even true outliers, reducing signal quality for business decisions.

In this pipeline, **UMAP + HDBSCAN** gives more robust and realistic theme discovery for unstructured B2B feedback.

---

## Repository File Guide

The table below documents every tracked file in this repository.

| Path | Purpose |
|---|---|
| `.gitignore` | Git ignore rules for environment and generated artifacts. |
| `.virtual_documents/data.ipynb` | Virtual notebook document artifact from local notebook tooling. |
| `anaconda_projects/db/project_filebrowser.db` | Local Anaconda/Jupyter file browser SQLite state. |
| `main.py` | Interactive orchestrator to build/load pipelines and trigger query → clustering → insights. |
| `data_input.py` | Input defaults, CSV loading, and validation of required/optional columns. |
| `cleaner.py` | Standardizes raw records into `title/content/metadata` and saves canonical CSV output. |
| `embedder.py` | Builds sentence embeddings from standardized text and stores vectors in `.npy` format. |
| `database.py` | Creates/manages ChromaDB persistent collection and upserts documents + metadata + vectors. |
| `analyzer.py` | Retrieval, UMAP/HDBSCAN clustering, cluster formatting, and Ollama insight generation logic. |
| `data.csv` | Raw sample/support dataset used for pipeline demonstrations. |
| `data/standardized_data.csv` | Canonical processed dataset ready for embedding/indexing. |
| `embeddings/embeddings.npy` | Precomputed embedding matrix for standardized records. |
| `data/llm_ready_clusters.json` | Clustered ticket groups serialized for LLM prompt input. |
| `data/final_b2b_insights.json` | Final generated business insights per discovered cluster. |
| `data/vector_store/0fc42724-a275-4260-bf69-d4f5cab679f3/data_level0.bin` | Chroma vector index binary segment. |
| `data/vector_store/0fc42724-a275-4260-bf69-d4f5cab679f3/header.bin` | Chroma vector index header metadata. |
| `data/vector_store/0fc42724-a275-4260-bf69-d4f5cab679f3/index_metadata.pickle` | Chroma index metadata/state for persisted collection. |
| `data/vector_store/0fc42724-a275-4260-bf69-d4f5cab679f3/length.bin` | Chroma index vector length metadata file. |
| `data/vector_store/0fc42724-a275-4260-bf69-d4f5cab679f3/link_lists.bin` | Chroma HNSW link graph file for ANN retrieval. |
| `data_input.ipynb` | Notebook walkthrough for raw CSV loading and schema normalization preparation. |
| `embeddings.ipynb` | Notebook for embedding model loading and vector generation experiments. |
| `vector_database.ipynb` | Notebook for Chroma collection creation and embedding ingestion flow. |
| `query.ipynb` | Notebook for semantic retrieval, filtering, and clustering exploration. |
| `llm.ipynb` | Notebook for cluster-to-insight generation using local Ollama models. |
| `notebooks/01_environment_check.ipynb` | Environment/package readiness checks and quick dataset sanity inspection. |

---

## Notebook Summaries

### `notebooks/01_environment_check.ipynb`
Validates Python/NLP stack readiness (pandas, NumPy, torch, transformers, sentence-transformers, sklearn, UMAP, Chroma, plotting libs) and performs quick CSV inspection.

### `data_input.ipynb`
Demonstrates loading raw input data and preparing records for downstream structured processing.

### `embeddings.ipynb`
Demonstrates converting cleaned text rows into numerical vectors (including FAISS-related experimentation utilities).

### `vector_database.ipynb`
Shows how embeddings and text/metadata records are inserted into a vector database collection.

### `query.ipynb`
Explores semantic search behavior, retrieved record inspection, and clustering-oriented analysis flow.

### `llm.ipynb`
Demonstrates prompt construction and LLM inference to transform clustered records into business-facing recommendations.

---

## Environment Setup

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Install and run Ollama (required for insight generation)

- Install Ollama from: https://ollama.com/
- Pull the default model used in code:

```bash
ollama pull llama3:latest
```

- Ensure the Ollama service is running before insight generation.

---

## `requirements.txt` (recommended content)

```txt
pandas
numpy
chromadb
sentence-transformers
umap-learn
hdbscan
ollama
faiss-cpu
scikit-learn
jsonschema
ipython
notebook
beautifulsoup4
emoji
torch
transformers
matplotlib
seaborn
```

---

## Usage: Run the Full Pipeline and Generate Insights

### Option A: Interactive full pipeline

```bash
python main.py
```

Then:
1. Choose **Build a New Dataset Pipeline**.
2. Provide the raw CSV path (for example `data.csv`).
3. Map title/content/metadata columns in the CLI menu.
4. Wait for standardization, embeddings, and vector DB indexing.
5. Enter a business query (for example, `refund delays`, `battery issue`, `late delivery`).
6. Review generated cluster and insight JSON outputs.

### Option B: Run modules step-by-step

```bash
python cleaner.py
python embedder.py
python database.py
python analyzer.py
```

---

## Output Artifacts

- Standardized records: `data/standardized_data.csv`
- Embeddings: `embeddings/embeddings.npy`
- Persisted vector index: `data/vector_store/...`
- Cluster payload for LLM: `data/llm_ready_clusters.json`
- Final business insights: `data/final_b2b_insights.json`

---

## Contributor Notes

- Keep raw datasets versioned carefully; avoid committing sensitive customer information.
- Rebuild embeddings and vector index when changing text fields or embedding model.
- Treat LLM outputs as decision-support signals and validate with domain stakeholders.
