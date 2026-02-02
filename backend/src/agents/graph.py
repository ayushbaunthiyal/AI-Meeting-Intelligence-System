"""
LangGraph Orchestration for Meeting Analysis

This module defines the LangGraph state graph that orchestrates
the meeting analysis pipeline with multiple nodes.
"""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes import actions_node, decisions_node, parser_node, summarizer_node
from .state import AgentState, create_initial_state

logger = logging.getLogger(__name__)


def create_analysis_graph() -> StateGraph:
    """
    Create the meeting analysis graph.
    
    The graph flows:
    Parser -> Summarizer -> Decisions -> Actions
    
    Each node processes the transcript and adds its analysis
    to the shared state.
    
    Returns:
        Compiled StateGraph for meeting analysis
    """
    # Create the graph with AgentState
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("parser", parser_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("decisions", decisions_node)
    workflow.add_node("actions", actions_node)
    
    # Define the flow
    # START -> Parser -> Summarizer -> Decisions -> Actions -> END
    workflow.add_edge(START, "parser")
    workflow.add_edge("parser", "summarizer")
    workflow.add_edge("summarizer", "decisions")
    workflow.add_edge("decisions", "actions")
    workflow.add_edge("actions", END)
    
    # Compile the graph
    return workflow.compile()


class MeetingAnalyzer:
    """
    High-level interface for meeting analysis.
    
    Uses the LangGraph pipeline to analyze meeting transcripts
    and extract summaries, decisions, and action items.
    """
    
    def __init__(self) -> None:
        """Initialize the meeting analyzer with the analysis graph."""
        self._graph = create_analysis_graph()
        logger.info("Meeting analyzer initialized")
    
    async def analyze(
        self,
        meeting_id: str,
        meeting_title: str,
        raw_transcript: str,
    ) -> dict[str, Any]:
        """
        Analyze a meeting transcript.
        
        Runs the full analysis pipeline:
        1. Parse and normalize the transcript
        2. Generate a summary
        3. Extract decisions
        4. Identify action items
        
        Args:
            meeting_id: Unique identifier for the meeting
            meeting_title: Title of the meeting
            raw_transcript: Raw transcript text
        
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Starting analysis for meeting: {meeting_id}")
        
        # Create initial state
        initial_state = create_initial_state(
            meeting_id=meeting_id,
            meeting_title=meeting_title,
            raw_transcript=raw_transcript,
        )
        
        # Run the graph
        final_state = await self._graph.ainvoke(initial_state)
        
        logger.info(f"Analysis complete for meeting: {meeting_id}")
        
        return {
            "meeting_id": meeting_id,
            "meeting_title": meeting_title,
            "participants": final_state.get("participants", []),
            "summary": final_state.get("summary"),
            "key_topics": final_state.get("key_topics", []),
            "decisions": final_state.get("decisions", []),
            "action_items": final_state.get("action_items", []),
            "parsed_transcript": final_state.get("parsed_transcript", ""),
            "error": final_state.get("error"),
        }
    
    def analyze_sync(
        self,
        meeting_id: str,
        meeting_title: str,
        raw_transcript: str,
    ) -> dict[str, Any]:
        """
        Synchronous version of analyze.
        
        Args:
            meeting_id: Unique identifier for the meeting
            meeting_title: Title of the meeting
            raw_transcript: Raw transcript text
        
        Returns:
            Dictionary containing analysis results
        """
        import asyncio
        return asyncio.run(self.analyze(meeting_id, meeting_title, raw_transcript))


# Singleton instance
_analyzer: MeetingAnalyzer | None = None


def get_meeting_analyzer() -> MeetingAnalyzer:
    """
    Get the meeting analyzer instance.
    
    Returns:
        MeetingAnalyzer: Singleton instance
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = MeetingAnalyzer()
    return _analyzer
