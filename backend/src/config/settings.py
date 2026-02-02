"""
Application Configuration using Pydantic Settings

This module centralizes all configuration for the Meeting Intelligence System.
Configuration values are loaded from environment variables with sensible defaults.

=============================================================================
WHY PYDANTIC SETTINGS?
=============================================================================

1. TYPE SAFETY: Values are validated at startup (not at runtime)
2. ENVIRONMENT VARIABLES: Automatic loading from .env files
3. DOCUMENTATION: Type hints serve as documentation
4. DEFAULTS: Easy to specify sensible defaults
5. VALIDATION: Custom validators for complex constraints

=============================================================================
CONFIGURATION HIERARCHY (lowest to highest priority):
=============================================================================

    1. Default values in this file (lowest priority)
    2. .env file (if present in project root)
    3. Environment variables (highest priority)

Example:
    # In .env file:
    OPENAI_API_KEY=sk-your-key-here
    WHISPER_MODEL=small
    
    # OR via environment:
    export OPENAI_API_KEY=sk-your-key-here

=============================================================================
IMPORTANT SETTINGS FOR NEW DEVELOPERS:
=============================================================================

REQUIRED:
    - OPENAI_API_KEY: Your OpenAI API key (get from platform.openai.com)

OPTIONAL (have defaults):
    - WHISPER_MODEL: Size of Whisper model (tiny/base/small/medium/large)
    - OPENAI_MODEL: Chat model to use (gpt-3.5-turbo/gpt-4)
    - LOG_LEVEL: Logging verbosity (DEBUG/INFO/WARNING/ERROR)

=============================================================================
"""

import logging
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings have reasonable defaults for local development.
    Only OPENAI_API_KEY is truly required for the app to function.
    
    USAGE:
    ------
        from src.config import get_settings
        
        settings = get_settings()
        print(settings.openai_model)  # "gpt-3.5-turbo"
    """
    
    # =========================================================================
    # Pydantic Settings Configuration
    # =========================================================================
    # This tells Pydantic how to load settings
    model_config = SettingsConfigDict(
        # Load from .env file in the current directory
        env_file=".env",
        # Also check parent directories for .env
        env_file_encoding="utf-8",
        # Don't fail if .env doesn't exist
        extra="ignore",
    )
    
    # =========================================================================
    # OpenAI Settings (REQUIRED for LLM functionality)
    # =========================================================================
    
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for LLM and embeddings. Get from platform.openai.com",
        # Environment variable name (automatic, but explicit for clarity)
    )
    
    openai_model: str = Field(
        default="gpt-3.5-turbo",
        description=(
            "OpenAI chat model to use. Options: "
            "gpt-3.5-turbo (fast, cheap), gpt-4 (smart, expensive)"
        ),
    )
    
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description=(
            "Embedding model for vector search. "
            "text-embedding-3-small is cost-effective with good quality."
        ),
    )
    
    # =========================================================================
    # Whisper Settings (Local Voice-to-Text)
    # =========================================================================
    # Whisper runs LOCALLY - no API costs! Model is downloaded on first use.
    
    whisper_model: str = Field(
        default="base",
        description=(
            "Whisper model size. Larger = more accurate but slower. "
            "Options: tiny (39M), base (74M), small (244M), medium (769M), large (1550M)"
        ),
    )
    
    # =========================================================================
    # Vector Store Settings (ChromaDB)
    # =========================================================================
    
    chroma_collection_name: str = Field(
        default="meeting_transcripts",
        description="Name of ChromaDB collection for storing embeddings",
    )
    
    chroma_persist_directory: str = Field(
        default="./data/chroma",
        description="Directory for ChromaDB persistence (not used in current in-memory mode)",
    )
    
    # =========================================================================
    # Text Chunking Settings (for RAG)
    # =========================================================================
    # These control how transcripts are split for embedding and retrieval
    
    chunk_size: int = Field(
        default=1000,
        description=(
            "Maximum characters per chunk. Smaller = more precise retrieval but more chunks. "
            "1000 is a good balance for meeting transcripts."
        ),
    )
    
    chunk_overlap: int = Field(
        default=200,
        description=(
            "Overlap between chunks. Helps ensure context isn't lost at boundaries. "
            "Should be ~20% of chunk_size."
        ),
    )
    
    # =========================================================================
    # Application Settings
    # =========================================================================
    
    backend_host: str = Field(
        default="0.0.0.0",
        description="Host to bind backend server (0.0.0.0 for Docker)",
    )
    
    backend_port: int = Field(
        default=8001,
        description="Port for backend API server",
    )
    
    ui_host: str = Field(
        default="0.0.0.0",
        description="Host to bind UI server",
    )
    
    ui_port: int = Field(
        default=8504,
        description="Port for Streamlit UI server",
    )
    
    # =========================================================================
    # Logging Settings
    # =========================================================================
    
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    
    log_format: str = Field(
        default="json",
        description="Log format: json (for production) or text (for development)",
    )
    
    # =========================================================================
    # Development Settings
    # =========================================================================
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode (more verbose logging, auto-reload)",
    )


# =============================================================================
# SINGLETON PATTERN FOR SETTINGS
# =============================================================================
# We use lru_cache to ensure settings are only loaded once.
# This is important because:
# 1. Loading settings involves file I/O (.env file)
# 2. Settings shouldn't change during runtime
# 3. Multiple modules import get_settings, so we want consistency

@lru_cache()
def get_settings() -> Settings:
    """
    Get the application settings (singleton).
    
    Uses lru_cache to ensure settings are loaded exactly once.
    All subsequent calls return the same cached instance.
    
    Returns:
        Settings: The application settings object
    
    Example:
        settings = get_settings()
        print(f"Using model: {settings.openai_model}")
    """
    logger.info("Loading application settings from environment")
    return Settings()
