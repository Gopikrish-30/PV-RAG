"""
Application Configuration Settings
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "PV-RAG"
    app_version: str = "1.0.0"
    debug: bool = True
    log_level: str = "INFO"
    api_port: int = 8000
    
    # Vector Store (Main Storage)
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "legal_rules"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # LLM - Groq (for response generation & web verification)
    groq_api_key: Optional[str] = None
    groq_model: str = "qwen/qwen3-32b"  # Fast and accurate
    use_llm_for_response: bool = True  # Enable LLM-powered responses
    
    # Web Search - Tavily
    tavily_api_key: Optional[str] = None
    
    # Dataset
    dataset_path: str = "./legal_dataset_extended_with_mods_20260205_210844.csv"
    dataset_cutoff_year: int = 2020
    
    # Web Verification (Optional)
    web_verification_enabled: bool = False  # Disabled by default
    web_verification_timeout: int = 10
    max_search_results: int = 5
    
    # Confidence
    min_confidence_score: float = 0.7
    high_confidence_threshold: float = 0.9
    
    # Temporal Query
    default_historical_cutoff_days: int = 365
    enable_timeline_generation: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
