from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tools as tools
import os
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from supabase_client import supabase
from data_processing import get_or_create_vector_db
from langchain_core.documents import Document
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import API_KEY_HEADER, APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

app = FastAPI(title="Dipin's AI Admin API")

api_key_header = APIKeyHeader(name="Dipin-Admin-Token", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == os.getenv("ADMIN_SECRET_KEY"):
        return api_key_header
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
    )

vector_db = get_or_create_vector_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your frontend's exact URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    from data_processing import sync_csv_to_chroma
    # This will pull everything from Supabase and put it in Chroma 
    # every time Render wakes the app up.
    sync_csv_to_chroma()
    
# schemas

class StatusUpdate(BaseModel):
    booking_id: str
    Status: str

class AnswerUpdate(BaseModel):
    query_id : str
    question : str
    answer : str
    category : str
    language : str
    keywords : str

# endpoints

@app.get("/api/pending-meetings", dependencies=[Depends(get_api_key)])
async def get_pending_meetings():
    response = supabase.table("pending_meetings").select("*").eq("status", "Pending").execute()
    
    formatted_meetings = []
    for row in response.data:
        formatted_meetings.append({
            "ID": row.get("id"),
            "Name": row.get("name"),
            "Email": row.get("email"),
            "Date": row.get("date"),
            "Time": row.get("time"),
            "Purpose": row.get("purpose"),
            "Status": row.get("status")
        })
        
    return formatted_meetings
    
@app.post("/api/approve-meeting", dependencies=[Depends(get_api_key)])
async def approve_meeting(update: StatusUpdate):

    response = supabase.table("pending_meetings").select("*").eq("id", update.booking_id).execute()

    if not response.data:
        raise HTTPException(status_code=404, detail="Meeting ID not found")
    
    meeting = response.data[0]
    result = tools.trigger_n8n_booking(
        meeting['name'], meeting['email'], meeting['date'], 
        meeting['time'], meeting['purpose']
    )

    if result.success:
        # 3. Update the CSV status
        tools.update_booking_status(update.booking_id, "Approved")
        return {"message": "Meeting approved and webhook triggered"}
    
    raise HTTPException(status_code=500, detail=result.message)
    

@app.post("/api/reject-meeting", dependencies=[Depends(get_api_key)])
async def reject_meeting(update: StatusUpdate):
    success = tools.update_booking_status(update.booking_id, "Rejected")

    if success:
        return {"message": "Meeting rejected"}
    
    return HTTPException(status_code=404, detail="Meeting ID not found")
    

@app.get("/api/queries", dependencies=[Depends(get_api_key)])
async def get_recent_queries():
    """
    Admin endpoint to fetch the latest user queries from the CSV.
    """
    try:

        response = supabase.table("required_updates").select("*")\
            .eq("status", "Pending")\
            .order("created_at", desc=True)\
            .limit(10).execute()
        
        recent_queries = [
            {"id": row.get("id"), "question": row.get("query"), "answer": row.get("status")}
            for row in response.data
        ]
        return recent_queries
    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        raise HTTPException(status_code=500, detail="Failed to read queries log")
    

@app.post("/api/answer-query", dependencies=[Depends(get_api_key)])
async def answer_query(update: AnswerUpdate):
    try:
        # 1. Update the required_updates table so it disappears from the dashboard
        supabase.table("required_updates").update({"status": "Answered"}).eq("id", update.query_id).execute()
        
        # 2. Insert the new knowledge into the mydetails table
        new_knowledge = {
            "topic": update.question,
            "content": update.answer,
            "category": update.category,
            "language": update.language,
            "keywords": update.keywords
        }
        supabase.table("mydetails").insert(new_knowledge).execute()
        
        new_doc = Document(
            page_content=str(update.answer),
            metadata={
                "topic": update.question,
                "category": update.category,
                "language": update.language,
                "keywords": update.keywords
            }
        )
        vector_db.add_documents([new_doc])

        return {"message": "Knowledge added to Supabase AND Vector DB trained successfully!"}
        
    except Exception as e:
        print(f"Error answering query: {e}")
        raise HTTPException(status_code=500, detail="Failed to save answer")


@app.get("/api/stats", dependencies=[Depends(get_api_key)])
async def get_stats():
    """
    Admin endpoint to fetch stats for the dashboard.
    """
    response = supabase.table("pending_meetings").select("status").execute()
    db_rows = response.data

    total_meetings = len(db_rows)
    approved_meetings = sum(1 for m in db_rows if m["status"] == "Approved")
    rejected_meetings = sum(1 for m in db_rows if m["status"] == "Rejected")
    
    processed_meetings = approved_meetings + rejected_meetings

    return {
        "total_requests": total_meetings,
        "processed_requests": processed_meetings
    }
    
