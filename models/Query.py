from pydantic import BaseModel
from typing import List



class Query(BaseModel):
    
    id: str
    prompt: str

class Query_Multiple(BaseModel):
    prompt: List[Query]


class SimilarPrompt(BaseModel):
    prompt: str
    distance: float  

class SearchResponse(BaseModel):
    results: List[SimilarPrompt]

class PromptVector(BaseModel):
    vector: int
    distance: float

class VectorResponse(BaseModel):
    results: List[PromptVector] 