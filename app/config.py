
"""
Configuration management for RAG system.
Handles environment variables and application settings.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # AWS Configuration
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    
    # Bedrock Models
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    EMBEDDING_MODEL_ID: str = "amazon.titan-embed-text-v1"
    
    # Model Parameters
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    LLM_TOP_P: float = 0.9
    
    # RAG Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Vector Store
    VECTOR_STORE_PATH: str = "data/processed/vector_store"
    EMBEDDING_DIMENSION: int = 1536
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_RELOAD: bool = False
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File Processing
    SUPPORTED_FORMATS: list = [".pdf", ".txt", ".docx", ".md"]
    MAX_FILE_SIZE_MB: int = 50
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()


# Create directories if they don't exist
def ensure_directories():
    """Create required directories if they don't exist."""
    settings = get_settings()
    settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (settings.BASE_DIR / "logs").mkdir(exist_ok=True)


ensure_directories()
