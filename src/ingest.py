import os
import logging
import faiss
import pickle
import numpy as np
import pandas as pd
from embed import get_embedding


# ===== PATH SETUP =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "transformed.csv")
DB_DIR = os.path.join(BASE_DIR, "db")

INDEX_PATH = os.path.join(DB_DIR, "index.faiss")
DOCS_PATH = os.path.join(DB_DIR, "docs.pkl")


# ===== LOGGING SETUP =====
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "pipeline.log")),
        logging.StreamHandler()
    ]
)


# ===== LOAD DATA =====
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Validate required column
    if "text" not in df.columns:
        raise ValueError("Missing 'text' column in transformed data")

    logging.info(f"Columns: {df.columns.tolist()}")
    logging.info(f"Total rows: {len(df)}")

    return df["text"].tolist()


# ===== MAIN PIPELINE =====
def main():
    try:
        logging.info("🚀 Starting embedding pipeline")

        documents = load_data()
        logging.info(f"Loaded {len(documents)} documents")

        logging.info("🧠 Generating embeddings...")
        embeddings = []
        total = len(documents)
        for i, doc in enumerate(documents, 1):
            embeddings.append(get_embedding(doc))
            if i % 10 == 0 or i == total:
                logging.info(f"  Embedded {i}/{total} documents")

        dim = len(embeddings[0])
        logging.info(f"Embedding dimension: {dim}")

        logging.info("📦 Creating FAISS index...")
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))

        os.makedirs(DB_DIR, exist_ok=True)

        logging.info("💾 Saving FAISS index...")
        faiss.write_index(index, INDEX_PATH)

        logging.info("💾 Saving documents...")
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(documents, f)

        logging.info("✅ Ingestion pipeline completed successfully")

    except Exception as e:
        logging.error(f"❌ Pipeline failed: {str(e)}")
        raise


# ===== RUN =====
if __name__ == "__main__":
    main()