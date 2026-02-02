"""
LangGraph Agent Orchestration - Main Graph Definition

This is the CORE of the meeting analysis system. It defines the LangGraph
pipeline that processes meeting transcripts through multiple AI-powered nodes.

=============================================================================
ARCHITECTURE OVERVIEW:
=============================================================================

The graph follows a LINEAR pipeline pattern:

    START
      │
      ▼
  ┌─────────┐    Normalizes transcript format, extracts speakers
  │ Parser  │    Input: raw_transcript
  │  Node   │    Output: parsed_transcript, participants
  └────┬────┘
       │
       ▼
  ┌─────────────┐    Generates high-level meeting overview
  │ Summarizer  │    Input: parsed_transcript
  │    Node     │    Output: summary, key_topics  
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐    Extracts decisions, agreements, pivots
  │  Decisions  │    Input: parsed_transcript
  │    Node     │    Output: decisions[]
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐    Identifies action items with owners/deadlines
  │  Actions    │    Input: parsed_transcript
  │    Node     │    Output: action_items[]
  └──────┬──────┘
         │
         ▼
       END

=============================================================================
KEY CONCEPTS:
=============================================================================

1. STATE MANAGEMENT:
   - LangGraph uses TypedDict to define state structure (see state.py)
   - Each node receives the full state and returns updates
   - Updates are MERGED into the existing state (not replaced)

2. ASYNC EXECUTION:
   - All nodes are async for better performance
   - Use `await` when calling the graph

3. NODE FUNCTIONS:
   - Receive `AgentState` as input
   - Return `dict` with state updates
   - Can optionally access LLM/embedding services

=============================================================================
USAGE EXAMPLE:
=============================================================================

    from src.agents import MeetingAnalyzer
    
    analyzer = MeetingAnalyzer()
    result = await analyzer.analyze(
        meeting_id="meeting-123",
        meeting_title="Product Roadmap Review",
        raw_transcript="[00:00] Sarah: Welcome everyone..."
    )
    
    print(result["summary"])       # High-level overview
    print(result["decisions"])     # List of Decision objects
    print(result["action_items"])  # List of ActionItem objects

=============================================================================
"""

import logging
from typing import Optional

from langgraph.graph import StateGraph, END

from .nodes import (
    actions_node,
    decisions_node,
    parser_node,
    summarizer_node,
)
from .state import AgentState

logger = logging.getLogger(__name__)


def create_analysis_graph() -> StateGraph:
    """
    Create the meeting analysis LangGraph workflow.
    
    This function builds and compiles the state graph that processes
    meeting transcripts through multiple analysis nodes.
    
    GRAPH STRUCTURE:
    ----------------
    The graph is a linear pipeline where each node processes the
    transcript in sequence:
    
    1. parser_node     → Normalizes format, extracts speakers
    2. summarizer_node → Generates meeting summary via LLM
    3. decisions_node  → Extracts decisions via LLM (JSON output)
    4. actions_node    → Extracts action items via LLM (JSON output)
    
    WHY LINEAR (not parallel)?
    --------------------------
    - Summarizer needs parsed transcript for better quality
    - Decisions/Actions could run in parallel, but sequential is simpler
    - LLM rate limits favor sequential execution anyway
    
    Returns:
        StateGraph: Compiled LangGraph that can be invoked with state
    """
    # ---------------------------------------------------------------------------
    # STEP 1: Create the graph builder with our state type
    # ---------------------------------------------------------------------------
    # StateGraph is parameterized with our AgentState TypedDict
    # This ensures type safety throughout the pipeline
    workflow = StateGraph(AgentState)
    
    # ---------------------------------------------------------------------------
    # STEP 2: Add all nodes to the graph
    # ---------------------------------------------------------------------------
    # Each node is an async function that:
    #   - Takes AgentState as input
    #   - Returns dict with state updates
    #
    # IMPORTANT: Node names are used for routing (see edges below)
    workflow.add_node("parser", parser_node)
    workflow.add_node("summarizer", summarizer_node)  
    workflow.add_node("decisions", decisions_node)
    workflow.add_node("actions", actions_node)
    
    # ---------------------------------------------------------------------------
    # STEP 3: Define the edges (execution order)
    # ---------------------------------------------------------------------------
    # set_entry_point: Where execution starts
    workflow.set_entry_point("parser")
    
    # add_edge: Connect nodes in sequence
    # Format: add_edge(from_node, to_node)
    workflow.add_edge("parser", "summarizer")
    workflow.add_edge("summarizer", "decisions")
    workflow.add_edge("decisions", "actions")
    
    # END: Special constant that marks the end of the graph
    workflow.add_edge("actions", END)
    
    # ---------------------------------------------------------------------------
    # STEP 4: Compile the graph
    # ---------------------------------------------------------------------------
    # compile() validates the graph and creates an executable workflow
    # The compiled graph can be invoked with: await graph.ainvoke(state)
    return workflow.compile()


class MeetingAnalyzer:
    """
    High-level interface for analyzing meeting transcripts.
    
    This class wraps the LangGraph pipeline and provides a simple
    API for running meeting analysis.
    
    USAGE:
    ------
        analyzer = MeetingAnalyzer()
        result = await analyzer.analyze(
            meeting_id="abc123",
            meeting_title="Sprint Planning",
            raw_transcript="[00:00] John: Let's start..."
        )
    
    ATTRIBUTES:
    -----------
        graph: The compiled LangGraph workflow
        
    METHODS:
    --------
        analyze(): Run full analysis on a transcript
    """
    
    def __init__(self) -> None:
        """
        Initialize the analyzer with a compiled graph.
        
        The graph is created once and reused for all analysis calls.
        This is efficient because graph compilation is relatively expensive.
        """
        logger.info("Initializing MeetingAnalyzer with LangGraph pipeline")
        self.graph = create_analysis_graph()
    
    async def analyze(
        self,
        meeting_id: str,
        meeting_title: str,
        raw_transcript: str,
    ) -> AgentState:
        """
        Analyze a meeting transcript through the full pipeline.
        
        This method:
        1. Creates initial state from inputs
        2. Runs the LangGraph pipeline
        3. Returns the final state with all analysis results
        
        Args:
            meeting_id: Unique identifier for the meeting (for tracking)
            meeting_title: Human-readable meeting title (used in LLM prompts)
            raw_transcript: The full transcript text to analyze
                           Format: "[timestamp] Speaker: Text" per line
        
        Returns:
            AgentState: Final state containing:
                - summary: str (LLM-generated overview)
                - key_topics: list[str] (extracted topics)
                - decisions: list[Decision] (extracted decisions)
                - action_items: list[ActionItem] (extracted tasks)
                - participants: list[str] (speaker names)
                - parsed_transcript: str (normalized format)
                - error: Optional[str] (if any step failed)
        
        Example:
            result = await analyzer.analyze(
                meeting_id="meeting-123",
                meeting_title="Q3 Planning",
                raw_transcript="[00:00] Alice: Welcome..."
            )
            
            for action in result["action_items"]:
                print(f"- {action.task} (Owner: {action.owner})")
        """
        logger.info(f"Starting analysis for meeting: {meeting_id}")
        
        # ---------------------------------------------------------------------------
        # Create initial state
        # ---------------------------------------------------------------------------
        # This is the input to the first node (parser)
        # Only the required fields are set; others will be populated by nodes
        initial_state: AgentState = {
            "meeting_id": meeting_id,
            "meeting_title": meeting_title,
            "raw_transcript": raw_transcript,
            # These will be populated by the pipeline nodes:
            "parsed_transcript": None,
            "participants": [],
            "summary": None,
            "key_topics": [],
            "decisions": [],
            "action_items": [],
            "error": None,
        }
        
        # ---------------------------------------------------------------------------
        # Run the graph
        # ---------------------------------------------------------------------------
        # ainvoke() runs the graph asynchronously
        # The state flows through: parser → summarizer → decisions → actions
        final_state = await self.graph.ainvoke(initial_state)
        
        logger.info(
            f"Analysis complete for meeting: {meeting_id}. "
            f"Found {len(final_state.get('decisions', []))} decisions, "
            f"{len(final_state.get('action_items', []))} action items"
        )
        
        return final_state
