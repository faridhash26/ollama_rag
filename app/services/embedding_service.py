import os
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-minilm:latest")

def ollama_embed(text: str):
    response = requests.post(OLLAMA_URL, json={"model": EMBEDDING_MODEL, "prompt": text})
    if response.status_code != 200:
        raise Exception("Failed to get embedding from Ollama")
    return response.json()["embedding"]

class OllamaEmbeddings:
    def embed_documents(self, texts):
        return [ollama_embed(text) for text in texts]

    def embed_query(self, text):
        return ollama_embed(text)
