"""
Q&A Agent - Retrieval-Augmented Generation (RAG) for Meeting Questions

This module implements a conversational Q&A agent that answers questions
about meeting transcripts using RAG (Retrieval-Augmented Generation).

=============================================================================
HOW RAG WORKS:
=============================================================================

Traditional LLM:
    Question → LLM → Answer (limited to training data)

RAG Approach:
    Question → Vector Search → Retrieve Relevant Chunks → LLM → Answer
    
    1. User asks: "What did Alice decide about the timeline?"
    2. Question is embedded into a vector
    3. Vector search finds similar chunks from the meeting transcript
    4. Relevant chunks are added to the LLM prompt as context
    5. LLM generates answer using ONLY the provided context

=============================================================================
WHY RAG?
=============================================================================

1. ACCURACY: Answers are grounded in actual meeting content
2. ATTRIBUTION: We can show which parts of the transcript were used
3. FRESHNESS: Can answer about meetings the LLM never saw during training
4. EFFICIENCY: Only sends relevant context, not entire transcript

=============================================================================
ARCHITECTURE:
=============================================================================

    User Question
         │
         ▼
    ┌─────────────────┐
    │  Vector Store   │  (ChromaDB)
    │  Similarity     │  Find chunks similar to question
    │  Search         │
    └────────┬────────┘
             │
             ▼
    Retrieved Context (top 5 chunks)
             │
             ▼
    ┌─────────────────┐
    │  LLM Service    │  (OpenAI GPT-3.5)
    │  Answer         │  Generate answer from context
    │  Generation     │
    └────────┬────────┘
             │
             ▼
    Answer + Source Citations

=============================================================================
"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..services import get_llm_service
from ..vectorstore import get_chroma_store

logger = logging.getLogger(__name__)


# =============================================================================
# SYSTEM PROMPT FOR Q&A
# =============================================================================
# This prompt is CRITICAL for RAG quality. Key principles:
# 1. Tell the LLM to ONLY use provided context (prevents hallucination)
# 2. Instruct to cite sources (accountability)
# 3. Allow "I don't know" (honesty when info isn't available)
# =============================================================================

QA_SYSTEM_PROMPT = """You are an AI assistant that answers questions about meeting transcripts.

CRITICAL RULES:
1. ONLY use information from the provided context to answer questions
2. If the answer is not in the context, say "I don't have that information"
3. Quote relevant parts of the transcript when helpful
4. Be concise but complete
5. If multiple speakers discussed the topic, mention their contributions

Context will be provided as excerpts from the meeting transcript.
Each excerpt includes timestamps and speaker names for reference."""


QA_USER_PROMPT = """Based on the following meeting context, answer the question.

--- MEETING CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Provide a clear, accurate answer based only on the context above."""


class QAAgent:
    """
    Conversational Q&A agent for meeting transcripts.
    
    This agent uses RAG to answer questions about meetings:
    1. Retrieves relevant transcript chunks from vector store
    2. Uses LLM to generate answers from the retrieved context
    3. Returns answers with source citations
    
    ATTRIBUTES:
    -----------
        llm_service: Service for LLM calls (chat completions)
        vector_store: ChromaDB store for semantic search
        
    USAGE:
    ------
        agent = QAAgent()
        result = await agent.ask(
            question="What was decided about the deadline?",
            meeting_id="meeting-123"
        )
        print(result["answer"])
        print(result["sources"])
    """
    
    def __init__(self) -> None:
        """
        Initialize the Q&A agent with required services.
        
        Both services are singletons (see their implementations),
        so they're shared across all QAAgent instances.
        """
        logger.info("Initializing QAAgent")
        self.llm_service = get_llm_service()
        self.vector_store = get_chroma_store()
    
    async def ask(
        self,
        question: str,
        meeting_id: Optional[str] = None,
        num_chunks: int = 5,
    ) -> dict:
        """
        Answer a question about meeting transcript(s).
        
        RAG PIPELINE:
        -------------
        1. Semantic search: Find relevant chunks using embeddings
        2. Context building: Combine chunks into prompt context
        3. LLM generation: Generate answer from context
        4. Source citation: Include which chunks were used
        
        Args:
            question: The user's natural language question
                     Examples: "What did Alice say about the timeline?"
                              "What are the action items from this meeting?"
            
            meeting_id: Optional filter to search only one meeting
                       If None, searches across ALL meetings in the store
            
            num_chunks: Number of context chunks to retrieve (default: 5)
                       Trade-off: More chunks = more context but higher cost
        
        Returns:
            dict containing:
                - answer: str - The LLM-generated answer
                - sources: list[str] - Transcript excerpts used as context
                - meeting_id: str - Which meeting was queried
        
        Example:
            result = await agent.ask(
                question="Who is responsible for the API changes?",
                meeting_id="sprint-planning-123"
            )
            # result = {
            #     "answer": "Bob is responsible for the API changes...",
            #     "sources": ["[00:15] Bob: I'll handle the API...", ...],
            #     "meeting_id": "sprint-planning-123"
            # }
        """
        logger.info(f"Answering question: {question[:50]}...")
        
        # -----------------------------------------------------------------------
        # STEP 1: Retrieve relevant context using semantic search
        # -----------------------------------------------------------------------
        # The vector store converts the question to an embedding and finds
        # the most similar chunks from the meeting transcript(s)
        #
        # HOW IT WORKS:
        # - Question → Embedding vector (via OpenAI embeddings)
        # - Compare against all stored transcript chunks
        # - Return top-K most similar (cosine similarity)
        documents = self.vector_store.search(
            query=question,
            meeting_id=meeting_id,
            k=num_chunks,
        )
        
        # -----------------------------------------------------------------------
        # STEP 2: Build context from retrieved documents
        # -----------------------------------------------------------------------
        # Combine all retrieved chunks into a single context string
        # Each chunk includes its position to help the LLM understand structure
        sources = []
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            content = doc.page_content
            sources.append(content)
            context_parts.append(f"[Excerpt {i}]: {content}")
        
        context = "\n\n".join(context_parts)
        
        # Handle case where no relevant context was found
        if not context:
            logger.warning("No relevant context found for question")
            return {
                "answer": "I couldn't find relevant information to answer that question.",
                "sources": [],
                "meeting_id": meeting_id,
            }
        
        # -----------------------------------------------------------------------
        # STEP 3: Generate answer using LLM
        # -----------------------------------------------------------------------
        # We use the chat completion API with:
        # - System message: Instructions for the LLM (rules, format)
        # - User message: The actual question with context
        user_message = QA_USER_PROMPT.format(
            context=context,
            question=question,
        )
        
        messages = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        
        # Call the LLM (async for non-blocking execution)
        answer = await self.llm_service.chat(messages)
        
        logger.info(f"Generated answer using {len(documents)} context chunks")
        
        # -----------------------------------------------------------------------
        # STEP 4: Return answer with sources
        # -----------------------------------------------------------------------
        # Including sources allows the UI to:
        # - Show users where the answer came from
        # - Build trust through transparency
        # - Let users verify the answer themselves
        return {
            "answer": answer,
            "sources": sources,
            "meeting_id": meeting_id,
        }


def get_qa_agent() -> QAAgent:
    """
    Factory function to get a QAAgent instance.
    
    NOTE: This creates a NEW instance each time (not a singleton).
    This is fine because the underlying services (LLM, Vector Store)
    are singletons, so we're not duplicating expensive resources.
    
    Returns:
        QAAgent: New Q&A agent instance
    """
    return QAAgent()
