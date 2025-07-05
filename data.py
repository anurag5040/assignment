from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model only once
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
doc_embeddings = []

def chunk_text(text: str, chunk_size: int = 2000) -> list:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def process_and_embed_text(text: str):
    global documents, doc_embeddings
    documents.clear()
    doc_embeddings.clear()

    # Chunk text
    chunks = chunk_text(text)
    documents.extend(chunks)

    # Compute and store embeddings
    doc_embeddings.extend(model.encode(chunks))

    print(f"✅ {len(chunks)} chunks created and embedded.")
    print(f"📐 Embedding dimension: {len(doc_embeddings[0]) if doc_embeddings else 0}")     

def get_most_similar(query: str, top_k: int = 3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "document": documents[idx],
            "similarity": round(float(similarities[idx]), 4)
        })
    return results
