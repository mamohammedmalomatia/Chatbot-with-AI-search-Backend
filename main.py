from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from rag_service import RAGService
import asyncio
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
import os

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service
rag_service = RAGService()

class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    search_type: str = "ai_search"  # ai_search, knowledge_graph, combined

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, str]]] = None
    search_type: str = "ai_search"  # ai_search, knowledge_graph, combined
    
class FollowUpRequest(BaseModel):
    query: str
    answer: str
    

class SuggestionsRequest(BaseModel):
    query: str

@app.post("/api/search")
async def search_documents(request: SearchRequest):
    """
    Search documents using the specified search type.
    """
    try:
        results = await rag_service._azure_search(
            query=request.query,
            filters=request.filters,
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Get AI response for chat queries.
    """
    try:
        response = await rag_service.get_chat_response(
            query=request.query,
            chat_history=request.chat_history,
            search_type=request.search_type
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/followups")
async def generate_followups(request: FollowUpRequest):
    try:
        questions = await rag_service._generate_follow_up_questions(request.query, request.answer)
        return {"follow_up_questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/suggestion")
async def suggestions(request: SuggestionsRequest):
    """
    Get AI response for chat queries.
    """
    try:
        response = await rag_service.get_suggestions(
            query=request.query
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

# Serve static files from the 'frontend' directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve index.html at root and for all unknown routes (for SPA support)
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    file_path = os.path.join("frontend", full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    return FileResponse(os.path.join("frontend", "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)