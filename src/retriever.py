import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search(query, top_k=3):
    index = faiss.read_index("corpus.index")
    with open("corpus.pkl", "rb") as f:
        sentences = pickle.load(f)

    query_embedding = model.encode([query])
    _, I = index.search(np.array(query_embedding), top_k)

    results = [sentences[i] for i in I[0]]
    return list(dict.fromkeys(results))  

def serper_search(query, num_results=3):
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    headers = { "X-API-KEY": SERPER_API_KEY }
    payload = { "q": query }

    res = requests.post("https://google.serper.dev/search", headers=headers, json=payload)
    results = res.json().get("organic", [])[:num_results]

    return [r['title'] + ": " + r['snippet'] for r in results]
