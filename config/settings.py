"""
Application Configuration Settings

NOTE: .env file values ALWAYS take priority over OS/shell environment variables.
This prevents stale exported env vars from overriding updated .env keys.
"""
from pydantic_settings import BaseSettings
from dotenv import dotenv_values
from typing import Optional
import os


def _load_env_overrides() -> dict:
    """Load .env file and return values that should override OS env vars."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        return {k: v for k, v in dotenv_values(env_path).items() if v is not None}
    return {}


class Settings(BaseSettings):
    """Application settings with environment variable support.
    
    Priority: .env file > OS environment variables > defaults
    """
    
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


def _create_settings() -> Settings:
    """Create Settings with .env values taking priority over OS env vars.
    
    By default, pydantic-settings gives OS env vars higher priority than .env.
    We flip that: temporarily inject .env values into os.environ so they win.
    """
    env_overrides = _load_env_overrides()
    saved = {}
    for key, val in env_overrides.items():
        saved[key] = os.environ.get(key)       # save original
        os.environ[key] = val                   # inject .env value
    
    s = Settings()
    
    # Restore original env (clean up)
    for key, original in saved.items():
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original
    
    return s


# Global settings instance
settings = _create_settings()


def reload_settings():
    """Reload settings from .env file (clears any cached/stale values).
    .env file values always take priority over OS environment variables.
    """
    global settings
    settings = _create_settings()
    return settings
