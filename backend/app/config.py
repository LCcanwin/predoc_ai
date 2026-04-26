"""Configuration for MiniMax API."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from backend directory
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# MiniMax API Configuration
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")

# CORS Configuration
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

# Vector store path
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", str(BASE_DIR / "storage" / "vector_db"))
DATA_PATH = os.getenv("DATA_PATH", str(BASE_DIR / "data" / "private"))


def get_llm_config() -> dict:
    """Get LLM configuration for LangChain."""
    return {
        "model": MINIMAX_MODEL,
        "api_key": MINIMAX_API_KEY,
        "base_url": MINIMAX_BASE_URL,
    }
