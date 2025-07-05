from playwright.sync_api import sync_playwright
import uvicorn
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query, HTTPException
from scraper import *
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from data import get_most_similar
import httpx
from configurations import collection
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class PromptRequest(BaseModel):
    prompt: str


@app.get("/search")
def search(query: str = Query(..., description="Search term")):
    try:
        links = get_top_5_links(query)
        print("links")
        print(links)
        save_links_to_docx(links)
        return {"query": query, "links": links, "message": "Embeddings generated successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    


@app.get("/")
def root():
    return {"message": "Welcome to the Sentence Embedding Search API!"}



# @app.post("/ask")
# async def ask_ollama(req: PromptRequest):
#     async with httpx.AsyncClient(timeout=120.0) as client:
#         response = await client.post("http://localhost:11434/api/generate", json={
#             "model": "gemma3:1b",   # change if you're using mistral, gemma etc.
#             "prompt": req.prompt,
#             "stream": False
#         })
#         return {"response": response.json().get("response", "").strip()}
    

# Load sentence-transformer model once globally
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

from fastapi import APIRouter
from configurations import collection
from sentence_transformers.util import cos_sim
import numpy as np
import httpx

from fastapi import APIRouter
from sentence_transformers.util import cos_sim
import numpy as np
import torch

async def is_query_valid_llm(query: str) -> bool:
    prompt = (
        f"Is the following query meaningful in a normal human context? "
        f"Query: {query}\n\n"
        f"Respond with only YES or NO."
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post("http://localhost:11434/api/generate", json={
            "model": "gemma3:1b",  # Replace with your local model
            "prompt": prompt,
            "stream": False
        })

    result = response.json().get("response", "").strip().lower()
    print("=====")
    print(result)
    print("=====")
    return result.strip().lower().startswith("yes")


@app.post("/ask_with_context")
async def ask_with_context(req: QueryRequest):
    # Step 0: Let the LLM decide query validity
    is_valid = await is_query_valid_llm(req.query)
    print(is_valid)
    if not is_valid:
        return {
            "query": req.query,
            "response": "This is not a valid query."
        }

    # Step 1: Compute embedding of the incoming query
    query_embedding = embedding_model.encode(req.query)
    query_embedding_tensor = torch.tensor(query_embedding, dtype=torch.float32)

    # Step 2: Load all previous queries with embeddings from DB
    stored_docs = collection.find({}, {"embedding": 1, "response": 1, "query": 1})

    # Step 3: Compare cosine similarity
    for doc in stored_docs:
        if "embedding" not in doc or not isinstance(doc["embedding"], list):
            continue

        stored_embedding = torch.tensor(doc["embedding"], dtype=torch.float32)
        similarity = cos_sim(query_embedding_tensor, stored_embedding)[0][0].item()

        if similarity >= 0.70:
            return {
                "query": req.query,
                "response": doc["response"],
                "matched_with_query": doc.get("query", ""),
                "similarity": round(similarity, 4)
            }

    # Step 4: Get top-k most similar documents
    top_matches = get_most_similar(req.query, req.top_k)
    context = "\n\n".join([match["document"] for match in top_matches])

    # Step 5: Create prompt
    prompt = (
        f"Use the following information to answer the question:\n\n"
        f"{context}\n\n"
        f"Question: {req.query}\n"
        f"Answer:"
    )

    # Step 6: Ask LLM for final answer
    async with httpx.AsyncClient(timeout=120.0) as client:
        llm_response = await client.post("http://localhost:11434/api/generate", json={
            "model": "gemma3:1b",
            "prompt": prompt,
            "stream": False
        })

    final_response = llm_response.json().get("response", "").strip()

    # Step 7: Save query and response
    collection.insert_one({
        "query": req.query,
        "response": final_response,
        "embedding": query_embedding.tolist(),
        "top_context_docs": [match["document"] for match in top_matches]
    })

    return {
        "query": req.query,
        "response": final_response
    }


if __name__ == "__main__":
    # asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        reload=False,
        port=8002
    )