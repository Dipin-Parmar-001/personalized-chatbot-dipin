import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from data_processing import get_or_create_vector_db
from tools import trigger_n8n_booking, log_pending_booking
from supabase_client import supabase

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Setup
# ---------------------------------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5,
)

# ---------------------------------------------------------------------------
# RAG prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the official AI Assistant for Dipen Parmar. Your goal is to represent him "
    "professionally and provide accurate information based on the context provided."
    "\n\n"
    "--- CORE RULES ---\n"
    "1. IDENTITY: Always assume that the retrieved context provided refers directly to Dipen Parmar. "
    "Answer in the third person or as his representative.\n"
    "2. RAG LOGIC: Use the 'Context' section to answer questions. If the context is empty or "
    "irrelevant to the question, you MUST say exactly: 'I am sorry, but I don't have that "
    "information about Dipen yet. I have noted it down for him to update me later.'\n"
    "3. NO HALLUCINATION: Do not make up facts. Only use what is in the context.\n"
    "\n"
    "--- BOOKING & LEAD LOGIC ---\n"
    "4. COMMAND GUIDANCE: If a user expresses interest in hiring, meeting, or contacting Dipen, "
    "you must guide them to use the command-based booking system. "
    "Tell them: 'I'd be happy to help you schedule a meeting! Please reply starting with the "
    "command **/book** followed by your Name, Email, Date (YYYY-MM-DD), Time, and Purpose.'\n"
    "5. LEAD TRIGGER: If the intent is clearly a business inquiry but they haven't used "
    "the /book command yet, say: 'It sounds like you'd like to get in touch with Dipen. "
    "Please use the **/book** command followed by your Name, Email, Date (YYYY-MM-DD), "
    "Time, and Purpose to schedule a meeting.'\n"
    "\n\n"
    "Context:\n{context}"
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

# ---------------------------------------------------------------------------
# Extraction prompt & structured output
# ---------------------------------------------------------------------------

class MeetingDetails(BaseModel):
    name: Optional[str] = Field(None, description="The client's name")
    email: Optional[str] = Field(None, description="The client's email address")
    date: Optional[str] = Field(None, description="The preferred date in YYYY-MM-DD format")
    time: Optional[str] = Field(None, description="The preferred time in HH:MM 24-hour format")
    purpose: Optional[str] = Field(None, description="The purpose or topic of the meeting")


EXTRACTION_SYSTEM_PROMPT = (
    "You are a strict data extraction assistant. "
    "Your ONLY job is to extract meeting details from the text provided. "
    "CRITICAL RULES:\n"
    "- Do NOT invent, assume, or guess any data.\n"
    "- If a field is not explicitly mentioned, set it to null.\n"
    "- Dates must be converted to YYYY-MM-DD format.\n"
    "- Times must be converted to HH:MM 24-hour format (e.g., '11:00 AM' → '11:00', '3:30 PM' → '15:30').\n"
    "- Do not use placeholders like 'example@email.com' or 'TBD'."
)

extraction_llm = llm.with_structured_output(MeetingDetails)

extraction_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", EXTRACTION_SYSTEM_PROMPT),
            ("human", "{input}"),  # ← must match the key passed to .invoke()
        ]
    )
    | extraction_llm
)

# ---------------------------------------------------------------------------
# RAG response
# ---------------------------------------------------------------------------

def get_response(user_query: str, vector_db) -> str:
    """Retrieves relevant context and returns an LLM-generated answer."""
    results = vector_db.similarity_search_with_relevance_scores(user_query, k=20)

    if not results or results[0][1] < 0.5:
        log_missing_query(user_query)
        return (
            "I am sorry, I don't have that specific detail about Dipen yet, "
            "but I've asked him to update my database!"
        )

    context = "\n\n".join(
        f"Topic: {doc.metadata['topic']}\nContent: {doc.page_content}"
        for doc, _score in results
    )

    chain = rag_prompt | llm
    response = chain.invoke({"context": context, "question": user_query})
    return response.content


def log_missing_query(query: str) -> None:
    """Appends unanswered questions to a tracking CSV for future updates."""
    new_row = {
        "query": query,
        "status": "Pending"
    }
    try: 
        supabase.table("required_updates").insert(new_row).execute()
        logger.info("📌 Logged missing query: '%s'", query)
    except Exception as e:
        logger.error("Failed to log missing query: %s", e)


# ---------------------------------------------------------------------------
# Booking flow
# ---------------------------------------------------------------------------

def process_book_request(details_text: str) -> str:
    """
    Extracts meeting fields from the user's text and triggers the webhook.
    Returns a human-readable string in all cases.
    """
    logger.info("🔍 Extracting meeting details from: '%s'", details_text)

    try:
        # FIX: pass a dict with key "input" to match the extraction chain template
        details: MeetingDetails = extraction_chain.invoke({"input": details_text})
    except Exception as e:
        logger.error("Extraction failed: %s", e)
        return (
            "I had trouble reading your booking details. "
            "Please try again using the format: "
            "**/book** Name, Email, Date (YYYY-MM-DD), Time, Purpose."
        )

    missing_fields = []
    if not details.name:
        missing_fields.append("Name")
    if not details.email:
        missing_fields.append("Email")
    if not details.date:
        missing_fields.append("Date (YYYY-MM-DD)")
    if not details.time:
        missing_fields.append("Time")

    if missing_fields:
        return (
            f"I'm almost ready to book that! I just need your: "
            f"**{', '.join(missing_fields)}**."
        )

    logger.info("Extracted details: %s", details)

    result = log_pending_booking(
        name=details.name,
        email=details.email,
        date=details.date,
        time=details.time,
        purpose=details.purpose or "General Discussion",
    )

    if result.success:
        return (
            f"📩 **Request Received!**\n\n"
            f"I've sent your meeting request to Dipen for approval. "
            f"Once he confirms the time, you'll receive a notification at **{details.email}**.\n\n"
            f"**Details sent for approval:**\n"
            f"- **Date:** {details.date}\n"
            f"- **Time:** {details.time}"
        )
    else:
        # result.message already contains a user-friendly error from tools.py
        return result.message


# ---------------------------------------------------------------------------
# Main agent router
# ---------------------------------------------------------------------------

def main_agent(user_input: str, vector_db) -> str:
    user_input = user_input.strip()

    # FIX: use removeprefix for clean, correct stripping of the command keyword
    if user_input.lower().startswith("/book"):
        details_text = user_input[len("/book"):].strip()
        if not details_text:
            return (
                "It looks like you used the /book command but didn't include your details. "
                "Please provide: **/book** Name, Email, Date (YYYY-MM-DD), Time, Purpose."
            )
        return process_book_request(details_text)

    # Legacy support for "my name is ..." style messages
    if user_input.lower().startswith("my name is"):
        return process_book_request(user_input)

    return get_response(user_input, vector_db)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading AI Brain...")
    my_db = get_or_create_vector_db()

    # test_input = (
    #    "/book I want to book a meeting with Dipen on 27-04-2026 for a discussion about "
    #    "collaborating on a new GenAI project. My email is chirag962469@gmail.com and "
    #    "I want to book it at 11:00 AM. My name is Chirag."
    #)

    test_input = input("You: ")
    ans = main_agent(test_input, my_db)
    print(ans)
