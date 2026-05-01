import logging
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from supabase_client import supabase

load_dotenv()

logger = logging.getLogger(__name__)

embeddings = MistralAIEmbeddings()
PERSIST_DIRECTORY = "./chroma_db"


def get_or_create_vector_db() -> Chroma:
    """
    Initialises the Chroma vector store, creating it on disk if it doesn't
    exist yet. Safe to call multiple times — Chroma deduplicates internally.
    """
    if not os.path.exists(PERSIST_DIRECTORY):
        logger.info("📂 Creating new Chroma storage at '%s'...", PERSIST_DIRECTORY)
    else:
        logger.info("📁 Loading existing Chroma storage from '%s'...", PERSIST_DIRECTORY)

    vector_db = Chroma(
        collection_name="my_collection",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    return vector_db


def sync_csv_to_chroma(csv_path: str = None) -> Chroma:
    """
    One-time sync: reads a CSV and upserts its rows into Chroma.
    Skips the sync if data already exists (idempotent).

    Expected CSV columns: Topic, Content, Category, Language
    """
    vector_db = get_or_create_vector_db()

    existing_count = vector_db._collection.count()
    if existing_count > 0:
        logger.info(
            "✨ Data already exists (%d entries). Skipping sync.", existing_count
        )
        return vector_db

    response = supabase.table("mydetails").select("*").execute()
    data = response.data
    
    if not data:
        logger.warning("No data found in Supabase table 'mydetails'.")
        return vector_db
    
    docs = [
        Document(
            page_content=str(row.get("content", "")),
            metadata={
                "topic": row.get("topic", ""),
                "category": row.get("category", ""),
                "language": row.get("language", ""),
            },
        )
        for row in data
    ]

    vector_db.add_documents(docs)
    logger.info("✅ Successfully synced %d entries to Chroma.", len(docs))
    return vector_db


# ---------------------------------------------------------------------------
# FIX: Removed the module-level `sync_csv_to_chroma("mydetail.csv")` call.
# Importing this module no longer triggers any side effects.
# Call sync_csv_to_chroma() explicitly from your setup/migration script:
#
#   from data_processing import sync_csv_to_chroma
#   sync_csv_to_chroma("mydetail.csv")
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run this file directly to do a one-time CSV → Chroma sync
    sync_csv_to_chroma("mydetail.csv")
