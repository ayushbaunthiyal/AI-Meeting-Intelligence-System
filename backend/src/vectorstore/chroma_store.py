"""
ChromaDB Vector Store for Meeting Transcripts

This module provides in-memory vector storage using ChromaDB
for semantic search over meeting transcripts.
"""

import logging
import uuid
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_settings
from ..models import Meeting, TranscriptSegment
from ..services import get_embedding_service

logger = logging.getLogger(__name__)


class ChromaStore:
    """
    Vector store for meeting transcripts using ChromaDB.
    
    Provides semantic search capabilities for finding relevant
    parts of meeting transcripts based on natural language queries.
    
    Attributes:
        client: ChromaDB client instance
        collection: ChromaDB collection for transcripts
        vector_store: LangChain Chroma wrapper
    """
    
    _instance: Optional["ChromaStore"] = None
    _vector_store: Optional[Chroma] = None
    
    def __new__(cls) -> "ChromaStore":
        """Singleton pattern to reuse store instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the ChromaDB vector store."""
        if self._vector_store is None:
            settings = get_settings()
            
            logger.info("Initializing ChromaDB vector store (in-memory)")
            
            # Create ChromaDB client (in-memory for this implementation)
            self._client = chromadb.Client(
                ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            
            # Get embedding service
            embedding_service = get_embedding_service()
            
            # Initialize LangChain Chroma wrapper
            self._vector_store = Chroma(
                client=self._client,
                collection_name=settings.chroma_collection_name,
                embedding_function=embedding_service.embeddings,
            )
            
            # Initialize text splitter
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            
            logger.info("ChromaDB vector store initialized successfully")
    
    @property
    def vector_store(self) -> Chroma:
        """Get the Chroma vector store."""
        if self._vector_store is None:
            raise RuntimeError("Vector store not initialized")
        return self._vector_store
    
    def add_meeting(self, meeting: Meeting) -> int:
        """
        Add a meeting transcript to the vector store.
        
        Args:
            meeting: Meeting object with transcript data
        
        Returns:
            Number of chunks added to the store
        """
        logger.info(f"Adding meeting to vector store: {meeting.id}")
        
        # Create documents from transcript segments
        documents = []
        
        # If we have segments, create documents from them
        if meeting.segments:
            for i, segment in enumerate(meeting.segments):
                doc = Document(
                    page_content=f"[{segment.timestamp}] {segment.speaker}: {segment.text}",
                    metadata={
                        "meeting_id": meeting.id,
                        "meeting_title": meeting.title,
                        "speaker": segment.speaker,
                        "timestamp": segment.timestamp,
                        "segment_index": i,
                        "source": "segment",
                    },
                )
                documents.append(doc)
        
        # If we have raw transcript, chunk and add it
        if meeting.raw_transcript:
            chunks = self._text_splitter.split_text(meeting.raw_transcript)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "meeting_id": meeting.id,
                        "meeting_title": meeting.title,
                        "chunk_index": i,
                        "source": "raw_transcript",
                    },
                )
                documents.append(doc)
        
        if not documents:
            logger.warning(f"No documents to add for meeting: {meeting.id}")
            return 0
        
        # Add documents to vector store
        self.vector_store.add_documents(documents)
        
        logger.info(f"Added {len(documents)} chunks for meeting: {meeting.id}")
        return len(documents)
    
    def add_transcript(
        self,
        meeting_id: str,
        title: str,
        segments: list[TranscriptSegment],
    ) -> int:
        """
        Add transcript segments to the vector store.
        
        Args:
            meeting_id: Unique identifier for the meeting
            title: Meeting title
            segments: List of transcript segments
        
        Returns:
            Number of chunks added
        """
        meeting = Meeting(
            id=meeting_id,
            title=title,
            segments=segments,
            participants=list({seg.speaker for seg in segments}),
        )
        return self.add_meeting(meeting)
    
    def search(
        self,
        query: str,
        meeting_id: Optional[str] = None,
        k: int = 5,
    ) -> list[Document]:
        """
        Search for relevant transcript chunks.
        
        Args:
            query: Natural language search query
            meeting_id: Optional filter by meeting ID
            k: Number of results to return
        
        Returns:
            List of relevant documents
        """
        logger.debug(f"Searching for: {query}")
        
        filter_dict = None
        if meeting_id:
            filter_dict = {"meeting_id": meeting_id}
        
        results = self.vector_store.similarity_search(
            query,
            k=k,
            filter=filter_dict,
        )
        
        logger.debug(f"Found {len(results)} results")
        return results
    
    def search_with_scores(
        self,
        query: str,
        meeting_id: Optional[str] = None,
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Search with relevance scores.
        
        Args:
            query: Natural language search query
            meeting_id: Optional filter by meeting ID
            k: Number of results to return
        
        Returns:
            List of (document, score) tuples
        """
        filter_dict = None
        if meeting_id:
            filter_dict = {"meeting_id": meeting_id}
        
        return self.vector_store.similarity_search_with_score(
            query,
            k=k,
            filter=filter_dict,
        )
    
    def get_retriever(
        self,
        meeting_id: Optional[str] = None,
        k: int = 5,
    ):
        """
        Get a retriever for use with LangChain.
        
        Args:
            meeting_id: Optional filter by meeting ID
            k: Number of results to return
        
        Returns:
            LangChain retriever instance
        """
        search_kwargs = {"k": k}
        if meeting_id:
            search_kwargs["filter"] = {"meeting_id": meeting_id}
        
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def delete_meeting(self, meeting_id: str) -> None:
        """
        Delete all chunks for a meeting.
        
        Args:
            meeting_id: ID of the meeting to delete
        """
        logger.info(f"Deleting meeting from vector store: {meeting_id}")
        
        # Get all document IDs for this meeting
        results = self.vector_store.get(where={"meeting_id": meeting_id})
        
        if results and results.get("ids"):
            self.vector_store.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for meeting: {meeting_id}")
        else:
            logger.warning(f"No documents found for meeting: {meeting_id}")
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        logger.warning("Clearing all documents from vector store")
        self._client.reset()
        
        # Reinitialize the collection
        settings = get_settings()
        embedding_service = get_embedding_service()
        self._vector_store = Chroma(
            client=self._client,
            collection_name=settings.chroma_collection_name,
            embedding_function=embedding_service.embeddings,
        )


def get_chroma_store() -> ChromaStore:
    """
    Get the ChromaDB store instance.
    
    Returns:
        ChromaStore: Singleton instance of the ChromaDB store
    """
    return ChromaStore()
