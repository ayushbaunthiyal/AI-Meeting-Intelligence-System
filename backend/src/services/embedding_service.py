"""
Embedding Service for AI Meeting Intelligence System

This service provides text embedding functionality using OpenAI's
embedding models for semantic search and retrieval.
"""

import logging
from typing import Optional

from langchain_openai import OpenAIEmbeddings

from ..config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings.
    
    Uses OpenAI's embedding models to convert text into vector
    representations for semantic search.
    
    Attributes:
        embeddings: The OpenAIEmbeddings instance
        model_name: Name of the embedding model being used
    """
    
    _instance: Optional["EmbeddingService"] = None
    _embeddings: Optional[OpenAIEmbeddings] = None
    
    def __new__(cls) -> "EmbeddingService":
        """Singleton pattern to reuse embeddings instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the embedding service with OpenAI."""
        if self._embeddings is None:
            settings = get_settings()
            self.model_name = settings.openai_embedding_model
            
            logger.info(f"Initializing embeddings with model: {self.model_name}")
            
            self._embeddings = OpenAIEmbeddings(
                model=self.model_name,
                api_key=settings.openai_api_key,
            )
            
            logger.info("Embedding service initialized successfully")
    
    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Get the OpenAIEmbeddings instance."""
        if self._embeddings is None:
            raise RuntimeError("Embeddings not initialized")
        return self._embeddings
    
    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            List of floats representing the embedding vector
        """
        return await self.embeddings.aembed_query(text)
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        return await self.embeddings.aembed_documents(texts)
    
    def embed_text_sync(self, text: str) -> list[float]:
        """
        Generate embedding for a single text (synchronous).
        
        Args:
            text: Text to embed
        
        Returns:
            List of floats representing the embedding vector
        """
        return self.embeddings.embed_query(text)
    
    def embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (synchronous).
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)


def get_embedding_service() -> EmbeddingService:
    """
    Get the embedding service instance.
    
    Returns:
        EmbeddingService: Singleton instance of the embedding service
    """
    return EmbeddingService()
