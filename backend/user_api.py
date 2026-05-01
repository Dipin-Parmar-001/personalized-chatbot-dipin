from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import agent as agent
import tools as tools
from data_processing import get_or_create_vector_db

app = FastAPI(title="Dipin's AI User API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change to your frontend URL in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (including the OPTIONS preflight)
    allow_headers=["*"],
)

vector_db = get_or_create_vector_db()

@app.on_event("startup")
async def startup_event():
    from data_processing import sync_csv_to_chroma
    # This will pull everything from Supabase and put it in Chroma 
    # every time Render wakes the app up.
    sync_csv_to_chroma()
    
# schemas

class QueryRequest(BaseModel):
    query: str
class BookingRequest(BaseModel):
    name: str
    email: str
    date: str
    time: str
    purpose: Optional[str] = "General Discussion"

# endpoints

@app.post("/api/ask")
async def ask_ai(request: QueryRequest):
    """
    User-facing RAG endpoint.
    Takes a question and returns the AI's response based on Dipin's context.
    """
    try:
        response = agent.main_agent(request.query, vector_db)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/book")
async def submit_booking(request: BookingRequest):
    """
    Endpoint for submitting a meeting request.
    This saves the request to the pending CSV for admin approval.
    """
    try:
        result = agent.log_pending_booking(
            name=request.name,
            email=request.email,
            date=request.date,
            time=request.time,
            purpose=request.purpose
        )

        if result.success:
            return {"status": "success", "message": result.message}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
