"""
ChromaDB Vector Store for Meeting Transcripts

This module provides semantic search over meeting transcripts using ChromaDB,
an open-source embedding database optimized for AI applications.

=============================================================================
WHAT IS A VECTOR STORE?
=============================================================================

A vector store is a database optimized for storing and searching embeddings
(high-dimensional vectors that represent the semantic meaning of text).

Traditional Search (keyword-based):
    Query: "project deadline"
    Matches: Documents containing "project" or "deadline"
    Problem: Misses "timeline", "due date", "delivery date"

Semantic Search (embedding-based):
    Query: "project deadline" → [0.23, -0.15, 0.87, ...]  (embedding)
    Matches: Documents with SIMILAR embeddings
    Finds: "timeline", "due date", "when is it due" (semantically similar)

=============================================================================
HOW CHROMADB WORKS:
=============================================================================

    1. TEXT INPUT
       "We decided to extend the deadline to Friday"
                           │
                           ▼
    2. EMBEDDING
       Text → OpenAI Embeddings API → [0.12, -0.34, 0.56, ...]
       (1536 dimensions for text-embedding-3-small)
                           │
                           ▼
    3. STORAGE
       ChromaDB stores:
       - id: unique identifier
       - embedding: the vector
       - document: original text
       - metadata: speaker, timestamp, meeting_id, etc.
                           │
                           ▼
    4. SEARCH
       Query embedding is compared to stored embeddings
       using cosine similarity. Top-K most similar returned.

=============================================================================
WHY CHROMADB?
=============================================================================

1. OPEN SOURCE: Free, no vendor lock-in
2. SIMPLE: No infrastructure needed, runs in-memory or persisted
3. PYTHON-NATIVE: Built for Python ML/AI workflows
4. LANGCHAIN INTEGRATION: Works seamlessly with LangChain

=============================================================================
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
    
    DESIGN PATTERN: Singleton
    -------------------------
    This class uses the Singleton pattern to ensure only one instance
    exists. This is important because:
    1. ChromaDB client manages connections efficiently when shared
    2. Embedding model is expensive to initialize
    3. We want consistent state across the application
    
    USAGE:
    ------
        from src.vectorstore import get_chroma_store
        
        # Get the singleton instance
        store = get_chroma_store()
        
        # Add a meeting
        store.add_meeting(meeting)
        
        # Search for relevant content
        results = store.search("What was discussed about the API?")
    """
    
    # =========================================================================
    # Singleton Implementation
    # =========================================================================
    _instance: Optional["ChromaStore"] = None
    _vector_store: Optional[Chroma] = None
    
    def __new__(cls) -> "ChromaStore":
        """
        Singleton pattern: Return existing instance or create new one.
        
        This ensures that no matter how many times you call ChromaStore(),
        you always get the same instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """
        Initialize the ChromaDB vector store.
        
        Only runs once (subsequent calls skip initialization because
        _vector_store is already set).
        """
        # Skip if already initialized
        if self._vector_store is not None:
            return
            
        settings = get_settings()
        
        logger.info("Initializing ChromaDB vector store (in-memory)")
        
        # ---------------------------------------------------------------------
        # Create ChromaDB Client
        # ---------------------------------------------------------------------
        # We use in-memory storage for this demo. For production, you'd
        # configure persistence to disk or use a hosted solution.
        self._client = chromadb.Client(
            ChromaSettings(
                # Disable anonymous usage tracking
                anonymized_telemetry=False,
                # Allow resetting the database (useful for testing)
                allow_reset=True,
            )
        )
        
        # ---------------------------------------------------------------------
        # Get Embedding Service
        # ---------------------------------------------------------------------
        # The embedding service wraps OpenAI's embedding API.
        # It converts text to vectors for storage and search.
        embedding_service = get_embedding_service()
        
        # ---------------------------------------------------------------------
        # Initialize LangChain Chroma Wrapper
        # ---------------------------------------------------------------------
        # LangChain provides a nice abstraction over ChromaDB that handles:
        # - Automatic text embedding
        # - Metadata filtering
        # - Retriever interface for RAG
        self._vector_store = Chroma(
            client=self._client,
            collection_name=settings.chroma_collection_name,
            embedding_function=embedding_service.embeddings,
        )
        
        # ---------------------------------------------------------------------
        # Initialize Text Splitter
        # ---------------------------------------------------------------------
        # For long transcripts, we split into chunks for better retrieval.
        # RecursiveCharacterTextSplitter tries to split at natural boundaries.
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,      # Max characters per chunk
            chunk_overlap=settings.chunk_overlap, # Overlap for context
            separators=["\n\n", "\n", ". ", " ", ""],  # Try in this order
        )
        
        logger.info("ChromaDB vector store initialized successfully")
    
    @property
    def vector_store(self) -> Chroma:
        """Get the underlying Chroma vector store."""
        if self._vector_store is None:
            raise RuntimeError("Vector store not initialized")
        return self._vector_store
    
    def add_meeting(self, meeting: Meeting) -> int:
        """
        Add a meeting transcript to the vector store.
        
        This method:
        1. Creates documents from transcript segments
        2. Chunks the raw transcript for additional coverage
        3. Embeds all documents using OpenAI
        4. Stores them in ChromaDB
        
        Args:
            meeting: Meeting object containing transcript data
        
        Returns:
            Number of chunks added to the store
        
        HOW DOCUMENTS ARE CREATED:
        --------------------------
        We create two types of documents:
        
        1. SEGMENT DOCUMENTS (from parsed segments):
           "[00:15] Alice: I think we should..." 
           - Preserves speaker and timestamp
           - Good for specific quotes
        
        2. CHUNKED DOCUMENTS (from raw transcript):
           - Splits long text into overlapping chunks
           - Good for general context
        """
        logger.info(f"Adding meeting to vector store: {meeting.id}")
        
        documents = []
        
        # ---------------------------------------------------------------------
        # Create documents from segments
        # ---------------------------------------------------------------------
        if meeting.segments:
            for i, segment in enumerate(meeting.segments):
                # Create LangChain Document with text and metadata
                doc = Document(
                    # The actual text content
                    page_content=f"[{segment.timestamp}] {segment.speaker}: {segment.text}",
                    # Metadata for filtering and attribution
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
        
        # ---------------------------------------------------------------------
        # Create documents from chunked raw transcript
        # ---------------------------------------------------------------------
        if meeting.raw_transcript:
            # Split into chunks
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
        
        # ---------------------------------------------------------------------
        # Add to vector store
        # ---------------------------------------------------------------------
        # This is where the magic happens:
        # 1. Each document's text is sent to OpenAI for embedding
        # 2. The embedding + document + metadata is stored in ChromaDB
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
        
        Convenience method that creates a Meeting object and calls add_meeting.
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
        
        This performs SEMANTIC SEARCH:
        1. Query is embedded using OpenAI
        2. Finds K most similar documents using cosine similarity
        3. Returns documents sorted by relevance
        
        Args:
            query: Natural language search query
                  Example: "What did they decide about the deadline?"
            
            meeting_id: Optional filter to search only one meeting.
                       If None, searches across ALL meetings.
            
            k: Number of results to return (default: 5)
               Trade-off: More results = more context but potentially less relevant
        
        Returns:
            List of Document objects, each containing:
            - page_content: The matching text
            - metadata: speaker, timestamp, meeting_id, etc.
        """
        logger.debug(f"Searching for: {query}")
        
        # Build filter for meeting_id if provided
        filter_dict = None
        if meeting_id:
            filter_dict = {"meeting_id": meeting_id}
        
        # Perform similarity search
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
        
        Same as search() but includes similarity scores.
        Scores range from 0 (least similar) to 1 (most similar).
        
        Useful for:
        - Filtering out low-relevance results
        - Debugging search quality
        - Display to users (e.g., "95% relevant")
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
        Get a LangChain retriever for use with RAG chains.
        
        The retriever interface is used by LangChain's RAG utilities.
        It provides a standard way to fetch relevant documents.
        """
        search_kwargs = {"k": k}
        if meeting_id:
            search_kwargs["filter"] = {"meeting_id": meeting_id}
        
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def delete_meeting(self, meeting_id: str) -> None:
        """
        Delete all chunks for a meeting.
        
        Removes all documents with the given meeting_id from ChromaDB.
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
        """
        Clear all documents from the vector store.
        
        WARNING: This deletes everything! Use with caution.
        """
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
    Get the ChromaDB store instance (singleton).
    
    Always returns the same instance, ensuring consistent state.
    
    Returns:
        ChromaStore: Singleton instance of the ChromaDB store
    """
    return ChromaStore()
