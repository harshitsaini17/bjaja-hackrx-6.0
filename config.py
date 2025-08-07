"""
Configuration settings for the LLM Document Processing System
"""
import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application settings with environment variable support"""
    
    # API Keys
    gemini_api_key: str = Field(default="", description="Gemini API Key")
    pinecone_api_key: str = Field(default="", description="Pinecone API Key")
    
    # Gemini Configuration
    gemini_model: str = "gemini-2.5-flash"
    gemini_pro_model: str = "gemini-2.5-pro"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    gemini_temperature: float = 0.1
    gemini_max_tokens: int = 8192
    
    # Pinecone Configuration
    pinecone_environment: str = "gcp-starter"
    pinecone_index_name: str = "insurance-policies"
    pinecone_dimension: int = 3072  # Gemini embedding dimension
    pinecone_metric: str = "cosine"
    
    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_page: int = 5
    
    # Search Configuration
    search_top_k: int = 10
    similarity_threshold: float = 0.7
    confidence_thresholds: Dict[str, float] = {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4
    }
    
    # File Processing
    supported_formats: List[str] = [".pdf", ".docx", ".txt"]
    max_file_size_mb: int = 50
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    
    # Data Directories
    data_dir: str = "sample_data"
    output_dir: str = "output"
    cache_dir: str = "cache"


def load_settings() -> Settings:
    """Load settings from environment variables"""
    from dotenv import load_dotenv
    load_dotenv()
    
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", "")
    )


# Global settings instance
settings = load_settings()
