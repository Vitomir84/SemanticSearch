import sys
import os
import copy
import uvicorn
import socket
import logging
import datetime
from models.prompt_search_engine import PromptSearchEngine
from models.data_reader import load_prompts_from_jsonl
from models.Query import Query, Query_Multiple, SearchResponse, SimilarPrompt, PromptVector, VectorResponse
from decouple import config
from fastapi import FastAPI, HTTPException, Depends, Body
from sentence_transformers import SentenceTransformer



prompt_path = r"C:\Users\jov2bg\Desktop\PromptSearch\search_engine\data\prompts_data.jsonl"


app = FastAPI(title="Search Prompt Engine", description="API for prompt search", version="1.0")

prompts = load_prompts_from_jsonl(prompt_path)
search_engine = PromptSearchEngine()
search_engine.add_prompts_to_vector_database(prompts)

@app.get("/")
def read_root():
    return {"message": "Prompt Search Engine is running!"}

@app.post("/search/")
async def search_prompts(query: Query, k: int = 3):
    print(f'Prompt: {query}')
    similar_prompts, distances = search_engine.most_similar(query.prompt, top_k=k)
    print(f'Similar Prompts {similar_prompts}')
    print(f'Distances {distances}')
    print(40*'****')
    # Format the response
    response = [
        SimilarPrompt(prompt=prompt, distance=float(distance)) 
        for prompt, distance in zip(similar_prompts, distances)
    ]
    
    return SearchResponse(results=response)

@app.post("/all_vectors_similarities/")
async def all_vectors(query: Query):

    query_embedding = search_engine.model.encode([query.prompt])  # Encode the prompt to a vector
    all_similarities = search_engine.cosine_similarity(query_embedding, search_engine.index)
    print(f'Prompt: {query}')
    print(f'All Vector Similarities: {all_similarities}')
    print(40*'****')
    response = [
        PromptVector(vector=index, distance=float(distance)) 
        for index, distance in enumerate(all_similarities)
    ]
    return VectorResponse(results=response)

if __name__ == "__main__":
    # Server Config
    # SERVER_HOST_IP = socket.gethostbyname(socket.gethostname())
    SERVER_HOST_IP = socket.gethostbyname("localhost") # for local deployment
    SERVER_PORT = int(8084)
    uvicorn.run(app, host=SERVER_HOST_IP, port=SERVER_PORT)