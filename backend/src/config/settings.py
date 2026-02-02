"""
AI Meeting Intelligence System - Backend Configuration

This module provides centralized configuration management using Pydantic Settings.
All configuration values are validated and loaded from environment variables.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings have sensible defaults for local development.
    For production, configure via environment variables or .env file.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # =========================================================================
    # OpenAI Configuration
    # =========================================================================
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for LLM and embedding models",
    )
    openai_model: str = Field(
        default="gpt-3.5-turbo",
        description="OpenAI model for chat completions",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI model for text embeddings",
    )
    
    # =========================================================================
    # Whisper Configuration
    # =========================================================================
    whisper_model: Literal["tiny", "base", "small", "medium", "large"] = Field(
        default="base",
        description="Whisper model size for voice-to-text",
    )
    
    # =========================================================================
    # Vector Store Configuration
    # =========================================================================
    chroma_persist_directory: str = Field(
        default="./data/chroma",
        description="Directory for ChromaDB persistence",
    )
    chroma_collection_name: str = Field(
        default="meeting_transcripts",
        description="ChromaDB collection name for transcripts",
    )
    
    # =========================================================================
    # Chunking Configuration
    # =========================================================================
    chunk_size: int = Field(
        default=1000,
        description="Size of text chunks for embedding",
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between text chunks",
    )
    
    # =========================================================================
    # Application Configuration
    # =========================================================================
    backend_host: str = Field(
        default="0.0.0.0",
        description="Backend API host",
    )
    backend_port: int = Field(
        default=8000,
        description="Backend API port",
    )
    
    # =========================================================================
    # Logging Configuration
    # =========================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: Literal["json", "console"] = Field(
        default="json",
        description="Log output format",
    )
    
    # =========================================================================
    # Development Settings
    # =========================================================================
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses LRU cache to ensure settings are only loaded once.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()
