"""
Agent State Definition for LangGraph

This module defines the state that flows through the LangGraph
agent orchestration pipeline.
"""

from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ..models import ActionItem, Decision, Meeting, MeetingSummary


class AgentState(TypedDict):
    """
    State that flows through the LangGraph agent.
    
    This state is passed between nodes and accumulates
    information as the meeting is analyzed.
    """
    
    # Input data
    meeting_id: str
    meeting_title: str
    raw_transcript: str
    parsed_transcript: str
    
    # Accumulated analysis results
    summary: Optional[str]
    decisions: list[Decision]
    action_items: list[ActionItem]
    key_topics: list[str]
    participants: list[str]
    
    # Chat messages for conversational queries
    messages: Annotated[list, add_messages]
    
    # Context from vector search
    context: str
    
    # Error tracking
    error: Optional[str]


class QueryState(TypedDict):
    """
    State for Q&A queries about meetings.
    """
    
    meeting_id: str
    question: str
    context: str
    chat_history: list[tuple[str, str]]
    answer: Optional[str]
    sources: list[str]


def create_initial_state(
    meeting_id: str,
    meeting_title: str,
    raw_transcript: str,
) -> AgentState:
    """
    Create initial state for meeting analysis.
    
    Args:
        meeting_id: Unique meeting identifier
        meeting_title: Title of the meeting
        raw_transcript: Raw transcript text
    
    Returns:
        Initial AgentState for the pipeline
    """
    return AgentState(
        meeting_id=meeting_id,
        meeting_title=meeting_title,
        raw_transcript=raw_transcript,
        parsed_transcript="",
        summary=None,
        decisions=[],
        action_items=[],
        key_topics=[],
        participants=[],
        messages=[],
        context="",
        error=None,
    )


def create_query_state(
    meeting_id: str,
    question: str,
    context: str = "",
    chat_history: Optional[list[tuple[str, str]]] = None,
) -> QueryState:
    """
    Create initial state for a Q&A query.
    
    Args:
        meeting_id: Meeting to query
        question: User's question
        context: Retrieved context from vector store
        chat_history: Previous conversation turns
    
    Returns:
        Initial QueryState
    """
    return QueryState(
        meeting_id=meeting_id,
        question=question,
        context=context,
        chat_history=chat_history or [],
        answer=None,
        sources=[],
    )
