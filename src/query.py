import faiss
import pickle
import numpy as np
import requests
from src.embed import get_embedding

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_DIR = os.path.join(BASE_DIR, "db")

INDEX_PATH = os.path.join(DB_DIR, "index.faiss")
DOC_PATH = os.path.join(DB_DIR, "docs.pkl")

def load_index():
    return faiss.read_index(INDEX_PATH)

def load_docs():
    with open(DOC_PATH, "rb") as f:
        return pickle.load(f)

def retrieve(query, index, documents, k=3):
    query_vec = np.array([get_embedding(query)]).astype("float32")
    D, I = index.search(query_vec, k)
    return [documents[i] for i in I[0]]

def generate_answer(query, context):
    prompt = f"""You are a telecom AI assistant.
Answer ONLY from the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{query}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()["response"]

def main():
    index = load_index()
    documents = load_docs()

    query = input("Ask your question: ")

    context_docs = retrieve(query, index, documents)
    context = "\n\n".join(context_docs)

    answer = generate_answer(query, context)

    print("\n💡 Answer:\n", answer)

if __name__ == "__main__":
    main()