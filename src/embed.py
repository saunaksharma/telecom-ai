import requests
import time

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "llama3:8b"
TIMEOUT = 60  # seconds


def get_embedding(text: str, retries: int = 3) -> list:
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": MODEL, "prompt": text},
                timeout=TIMEOUT
            )
            if response.status_code != 200:
                raise Exception(f"Ollama returned {response.status_code}: {response.text}")
            return response.json()["embedding"]
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)  # exponential back-off: 2s, 4s