import requests
import time
import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from supabase_client import supabase

load_dotenv()

logger = logging.getLogger(__name__)

N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

# Timeout for the webhook call (connect timeout, read timeout)
REQUEST_TIMEOUT = (5, 15)
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds between retries

import requests
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class BookingResult:
    success: bool
    message: str
    missing_fields: Optional[List[str]] = None

def trigger_n8n_booking(name, email, date, time, purpose):
    N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")
    payload = {"name": name, "email": email, "date": date, "time": time, "purpose": purpose}
    
    try:
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=15)
        
        if response.status_code == 200:
            data = response.json()

            # CASE 2: The Workflow finished successfully
            if data.get("status") == "workflow_complete":
                return BookingResult(success=True, message="Meeting booked successfully!")
            
            # CASE 3: Generic response (Workflow might be in 'test' mode or misconfigured)
            return BookingResult(False, "The system is online but didn't return a completion signal.")

        return BookingResult(False, f"Server error: {response.status_code}")

    except Exception as e:
        return BookingResult(False, f"Connection error: {str(e)}")
    

PENDING_FILE = "pending_meetings.csv"

def log_pending_booking(name, email, date, time, purpose):
    """Saves a meeting request for admin approval instead of triggering immediately."""
    new_request = {
        "name": name,
        "email": email,
        "date": date,
        "time": time,
        "purpose": purpose,
        "status": "Pending"
    }
    
    supabase.table("pending_meetings").insert(new_request).execute()

    return BookingResult(True, "Your request has been sent to Dipen for approval. You'll hear back soon!")

def update_booking_status(booking_id, new_status):
    """Updates the status of a booking request in the CSV file."""
    response = supabase.table("pending_meetings").update({"status": new_status}).eq("id", str(booking_id)).execute()
    
    return len(response.data) > 0
