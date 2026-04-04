"""Application settings loaded from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # Note: runtime code under app/core uses BASE_DIR-relative qdrant path; this is legacy / alternate.
    QDRANT_PATH = os.path.normpath(os.path.join(os.getcwd(), "qdrant_db"))
    COLLECTION_NAME = "ask_my_docs_collection"

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gemini-2.5-flash"

    RAW_PDF_DIR = "data/raw_pdfs"

settings = Settings()