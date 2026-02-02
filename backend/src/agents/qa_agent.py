"""
Q&A Agent for Meeting Queries

This module provides a conversational Q&A interface
for asking questions about meeting transcripts.
"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..services import get_llm_service
from ..vectorstore import get_chroma_store
from .state import QueryState, create_query_state

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = """You are a helpful meeting assistant. Your task is to answer questions 
about meeting transcripts based on the provided context.

Guidelines:
- Only use information from the provided context
- If the answer is not in the context, say "I couldn't find that information in the meeting transcript"
- Quote relevant parts of the transcript when appropriate
- Be concise but complete
- If asked about decisions, action items, or specific topics, focus on those
- Reference speakers by name when relevant

Context from the meeting transcript:
{context}
"""

QA_USER_PROMPT = """Question: {question}

Please answer based on the meeting transcript context provided above."""


class QAAgent:
    """
    Conversational Q&A agent for meeting transcripts.
    
    Uses RAG to retrieve relevant context from the vector store
    and generates answers using the LLM.
    """
    
    def __init__(self) -> None:
        """Initialize the Q&A agent."""
        self._llm_service = get_llm_service()
        self._vector_store = get_chroma_store()
        logger.info("Q&A Agent initialized")
    
    async def ask(
        self,
        question: str,
        meeting_id: str,
        chat_history: Optional[list[tuple[str, str]]] = None,
        k: int = 5,
    ) -> dict:
        """
        Ask a question about a meeting.
        
        Args:
            question: The user's question
            meeting_id: ID of the meeting to query
            chat_history: Optional previous conversation turns
            k: Number of context chunks to retrieve
        
        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Processing question for meeting {meeting_id}: {question[:50]}...")
        
        # Retrieve relevant context
        documents = self._vector_store.search(
            query=question,
            meeting_id=meeting_id,
            k=k,
        )
        
        if not documents:
            return {
                "answer": "I couldn't find any relevant information in this meeting transcript. "
                         "Please make sure the meeting has been uploaded and processed.",
                "sources": [],
            }
        
        # Build context from documents
        context = self._build_context(documents)
        sources = [doc.page_content for doc in documents]
        
        # Build messages
        system_prompt = QA_SYSTEM_PROMPT.format(context=context)
        user_prompt = QA_USER_PROMPT.format(question=question)
        
        messages = [SystemMessage(content=system_prompt)]
        
        # Add chat history if provided
        if chat_history:
            for role, content in chat_history[-5:]:  # Last 5 turns
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))
        
        messages.append(HumanMessage(content=user_prompt))
        
        # Generate answer
        answer = await self._llm_service.chat(messages)
        
        logger.info("Generated answer successfully")
        
        return {
            "answer": answer,
            "sources": sources[:3],  # Return top 3 sources
        }
    
    def _build_context(self, documents: list) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            documents: List of retrieved documents
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            speaker = metadata.get("speaker", "")
            timestamp = metadata.get("timestamp", "")
            
            if speaker and timestamp:
                context_parts.append(
                    f"[{i}] [{timestamp}] {speaker}:\n{doc.page_content}"
                )
            else:
                context_parts.append(f"[{i}] {doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def ask_sync(
        self,
        question: str,
        meeting_id: str,
        chat_history: Optional[list[tuple[str, str]]] = None,
        k: int = 5,
    ) -> dict:
        """
        Synchronous version of ask.
        
        Args:
            question: The user's question
            meeting_id: ID of the meeting to query
            chat_history: Optional previous conversation turns
            k: Number of context chunks to retrieve
        
        Returns:
            Dictionary with answer and sources
        """
        import asyncio
        return asyncio.run(self.ask(question, meeting_id, chat_history, k))


# Singleton instance
_qa_agent: QAAgent | None = None


def get_qa_agent() -> QAAgent:
    """
    Get the Q&A agent instance.
    
    Returns:
        QAAgent: Singleton instance
    """
    global _qa_agent
    if _qa_agent is None:
        _qa_agent = QAAgent()
    return _qa_agent
