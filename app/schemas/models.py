from pydantic import BaseModel
from typing import List, Dict

class ChatRequest(BaseModel):
    session_id: str
    question:   str

class ChatResponse(BaseModel):
    answer: str
    pages:  List[str]

class ResetRequest(BaseModel):
    session_id: str

class AnalyzeResponse(BaseModel):
    top_class:    str
    label:        str
    color:        str
    scores:       Dict[str, float]
    gpt_analysis: str
