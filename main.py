import os
import glob
import pandas as pd

# Import your custom modules
from cleaner import standardize_data
from embedder import generate_embeddings
from database import setup_vector_db
from analyzer import query_and_analyze


def get_existing_datasets():
    """Scans the processed_datasets directory for previously built pipelines."""
    base_dir = "processed_datasets"
    if not os.path.exists(base_dir):
        return []
    
    # List all folders inside processed_datasets
    datasets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return datasets


def interactive_column_mapping(df):
    """Interactive menu for column selection."""
    columns = list(df.columns)
    
    print("\n" + "=" * 50)
    print("📊 DATASET COLUMN MAPPING MENU")
    print("=" * 50)
    print("Found the following columns in your dataset:")
    for i, col in enumerate(columns):
        print(f"  [{i}] {col}")
    print("-" * 50)

    def get_indices(prompt_msg, allow_multiple=True, required=False):
        while True:
            val = input(prompt_msg).strip()
            if not val:
                if required:
                    print("❌ This field is required.")
                    continue
                return [] if allow_multiple else None
            
            try:
                indices = [int(x.strip()) for x in val.split(",")]
                selected_cols = [columns[idx] for idx in indices if 0 <= idx < len(columns)]
                
                if required and not selected_cols:
                    print("❌ No valid columns selected.")
                    continue
                    
                if not allow_multiple:
                    return selected_cols[0] if selected_cols else None
                return selected_cols

            except ValueError:
                print("❌ Invalid format. Please enter numbers separated by commas (e.g., 0,2).")

    title_column = get_indices("👉 Enter index for TITLE column (press Enter to skip): ", False, False)
    text_columns = get_indices("👉 Enter index(es) for CONTENT columns (comma-separated): ", True, True)
    metadata_columns = get_indices("👉 Enter index(es) for METADATA columns (comma-separated, Enter to skip): ", True, False)
    
    return text_columns, title_column, metadata_columns


def run_new_pipeline():
    """Builds a new database from scratch."""
    print("\n" + "=" * 50)
    print("🚀 INITIALIZING NEW DATA PIPELINE")
    print("=" * 50)
    
    raw_csv_path = input("Enter the path to your raw CSV file (e.g., data.csv): ").strip()

    if not os.path.exists(raw_csv_path):
        print(f"[ERROR] Could not find file at '{raw_csv_path}'")
        return None

    try:
        # Extract name to create a dedicated folder
        dataset_name = os.path.splitext(os.path.basename(raw_csv_path))[0]
        base_dir = os.path.join("processed_datasets", dataset_name)
        os.makedirs(base_dir, exist_ok=True)
        
        # Define the strict paths for this specific dataset
        clean_csv_path = os.path.join(base_dir, "1_standardized.csv")
        embeddings_path = os.path.join(base_dir, "2_embeddings.npy")
        vector_db_dir = os.path.join(base_dir, "3_vector_store")
        
        print(f"\n[INFO] Loading {raw_csv_path}...")
        raw_df = pd.read_csv(raw_csv_path)

        text_columns, title_column, metadata_columns = interactive_column_mapping(raw_df)

        print("\n[PHASE 1] Standardizing data...")
        standardize_data(
            df=raw_df,
            text_columns=text_columns,
            title_column=title_column,
            metadata_columns=metadata_columns,
            save_filename="1_standardized.csv",
            data_folder=base_dir
        )

        print("\n[PHASE 2] Generating Embeddings...")
        generate_embeddings(csv_path=clean_csv_path, embeddings_path=embeddings_path)
        
        print("\n[PHASE 3] Setting up Vector Database...")
        database_collection = setup_vector_db(
            csv_path=clean_csv_path, 
            embeddings_path=embeddings_path,
            persist_directory=vector_db_dir,
            collection_name=dataset_name
        )
        
        print("\n✅ New Pipeline built successfully!")
        return database_collection, base_dir
        
    except Exception as exc:
        print(f"\n[ERROR] Pipeline setup failed: {exc}")
        return None, None


def load_existing_pipeline(dataset_name):
    """Reconnects to a previously built ChromaDB collection."""
    print(f"\n[INFO] Reconnecting to existing database for '{dataset_name}'...")
    base_dir = os.path.join("processed_datasets", dataset_name)
    vector_db_dir = os.path.join(base_dir, "3_vector_store")
    
    if not os.path.exists(vector_db_dir):
        print(f"[ERROR] Vector database not found at {vector_db_dir}")
        return None, None
        
    try:
        import chromadb
        client = chromadb.PersistentClient(path=vector_db_dir)
        collection = client.get_collection(name=dataset_name)
        print(f"✅ Successfully loaded {collection.count()} records.")
        return collection, base_dir
    except Exception as e:
        print(f"[ERROR] Failed to load existing database: {e}")
        return None, None


# ==========================================
# --- The Main Orchestrator Loop ---
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("🧠 B2B INTELLIGENCE HUB")
    print("=" * 50)
    
    existing_datasets = get_existing_datasets()
    
    # 1. The Main Menu
    print("Select an option:")
    print("  [0] Build a New Dataset Pipeline")
    for i, dataset in enumerate(existing_datasets, 1):
        print(f"  [{i}] Load Existing: {dataset}")
        
    while True:
        try:
            choice = input("\nEnter your choice (number): ").strip()
            choice_idx = int(choice)
            
            if choice_idx == 0:
                database_collection, base_dir = run_new_pipeline()
                break
            elif 1 <= choice_idx <= len(existing_datasets):
                selected_dataset = existing_datasets[choice_idx - 1]
                database_collection, base_dir = load_existing_pipeline(selected_dataset)
                break
            else:
                print("❌ Invalid selection. Choose a number from the menu.")
        except ValueError:
            print("❌ Please enter a valid number.")

    # 2. The Interactive Search Loop
    if database_collection and base_dir:
        print("\n✅ System Ready for Queries.")
        
        # Define where the LLM files should be saved for this specific dataset
        clusters_path = os.path.join(base_dir, "4_llm_ready_clusters.json")
        insights_path = os.path.join(base_dir, "5_final_b2b_insights.json")
        
        while True:
            print("\n" + "-" * 50)
            user_query = input("Enter a topic to investigate (or type 'exit' to quit): ").strip()

            if user_query.lower() == "exit":
                print("Shutting down pipeline. Goodbye!")
                break

            if not user_query:
                print("[WARN] Please enter a query or type 'exit'.")
                continue
                
            print("\n[PHASE 4 & 5] Querying, Clustering, and Generating AI Insights...")
            query_and_analyze(
                query_text=user_query, 
                db_collection=database_collection,
                clusters_path=clusters_path,
                insights_path=insights_path
            )