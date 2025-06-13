from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
import pickle
import re

def split_into_chunks(text, max_chunk_len=1):
    sentences = re.split(r'\.\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return ['. '.join(sentences[i:i+max_chunk_len]) + '.' for i in range(0, len(sentences), max_chunk_len)]


def embed_corpus(file_path="/Users/snsp/Documents/rag-chatbot/corpus.csv", model_name="all-MiniLM-L6-v2"):
    df = pd.read_csv(file_path)

    raw_sentences = list(set(df["Sentence"].dropna().astype(str).tolist()))
    chunks = []
    for para in raw_sentences:
        para_chunks = split_into_chunks(para)
        chunks.extend(para_chunks)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))

    with open("corpus.pkl", "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, "corpus.index")

if __name__ == "__main__":
    embed_corpus()
