"""Vector store module for AI Meeting Intelligence System."""

from .chroma_store import ChromaStore, get_chroma_store

__all__ = [
    "ChromaStore",
    "get_chroma_store",
]
