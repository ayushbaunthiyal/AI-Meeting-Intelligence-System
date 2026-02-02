"""
LangGraph Agent State Definition

This module defines the state structure used by all LangGraph nodes.
In LangGraph, state is the shared data that flows through the graph.

=============================================================================
WHAT IS STATE IN LANGGRAPH?
=============================================================================

LangGraph uses a "state machine" pattern where:
1. Each node receives the FULL state as input
2. Each node returns a PARTIAL update (dict with changed fields)
3. Updates are MERGED into the existing state
4. The next node receives the updated state

Example flow:
    Initial State:
        {meeting_id: "123", raw_transcript: "...", summary: None}
                                │
                                ▼
    Parser Node returns: {parsed_transcript: "...", participants: ["A", "B"]}
                                │
                                ▼
    State after merge:
        {meeting_id: "123", raw_transcript: "...", summary: None,
         parsed_transcript: "...", participants: ["A", "B"]}
                                │
                                ▼
    Summarizer Node returns: {summary: "The meeting covered..."}
                                │
                                ▼
    State after merge:
        {meeting_id: "123", raw_transcript: "...", 
         parsed_transcript: "...", participants: ["A", "B"],
         summary: "The meeting covered..."}

=============================================================================
WHY TYPEDDICT?
=============================================================================

TypedDict (from typing module) provides:
1. TYPE HINTS: IDE autocomplete and type checking
2. DOCUMENTATION: Field types serve as documentation
3. VALIDATION: LangGraph can validate state structure
4. NO OVERHEAD: It's just a dict at runtime, no class overhead

We use TypedDict instead of Pydantic here because LangGraph expects
regular dicts for state management and merging.

=============================================================================
"""

from typing import Optional, TypedDict

from ..models import ActionItem, Decision


class AgentState(TypedDict, total=False):
    """
    State structure for the meeting analysis LangGraph.
    
    This TypedDict defines all fields that can exist in the state.
    Fields are organized by which node produces them.
    
    TOTAL=FALSE:
    ------------
    The `total=False` means all fields are OPTIONAL by default.
    This allows nodes to return partial updates without including
    every field. For example, the summarizer node only returns
    {summary: "...", key_topics: [...]} and that's valid.
    
    STATE FLOW:
    -----------
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                        INITIAL INPUT                             │
    │  meeting_id, meeting_title, raw_transcript                       │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      PARSER NODE OUTPUT                          │
    │  parsed_transcript, participants                                 │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    SUMMARIZER NODE OUTPUT                        │
    │  summary, key_topics                                             │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    DECISIONS NODE OUTPUT                         │
    │  decisions                                                       │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                     ACTIONS NODE OUTPUT                          │
    │  action_items                                                    │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    # =========================================================================
    # Input Fields (provided at graph invocation)
    # =========================================================================
    # These fields are set when the graph is first invoked.
    # They come from the user's request (e.g., uploaded transcript).
    
    meeting_id: str
    """Unique identifier for the meeting. Used for tracking and retrieval."""
    
    meeting_title: str
    """Human-readable title. Used in LLM prompts for context."""
    
    raw_transcript: str
    """
    The original, unprocessed transcript text.
    Various formats are supported:
    - [00:00] Speaker: Text
    - 00:00 - Speaker: Text
    - Speaker: Text (no timestamp)
    """
    
    # =========================================================================
    # Parser Node Output
    # =========================================================================
    # The parser node normalizes the transcript format and extracts metadata.
    
    parsed_transcript: Optional[str]
    """
    Normalized transcript in consistent format: [timestamp] Speaker: Text
    Each line follows the same structure for easier processing.
    """
    
    participants: list[str]
    """
    List of unique speaker names extracted from the transcript.
    Example: ["Alice", "Bob", "Carol"]
    """
    
    # =========================================================================
    # Summarizer Node Output
    # =========================================================================
    # The summarizer uses an LLM to generate a high-level overview.
    
    summary: Optional[str]
    """
    LLM-generated meeting summary.
    Typically 3-5 paragraphs covering main topics and outcomes.
    """
    
    key_topics: list[str]
    """
    List of main topics discussed.
    Example: ["Budget Review", "Q3 Timeline", "Team Hiring"]
    """
    
    # =========================================================================
    # Decisions Node Output
    # =========================================================================
    # The decisions node extracts key decisions using an LLM.
    
    decisions: list[Decision]
    """
    List of Decision objects extracted from the meeting.
    Each decision includes what was decided, who made it, and context.
    """
    
    # =========================================================================
    # Actions Node Output
    # =========================================================================
    # The actions node extracts action items using an LLM.
    
    action_items: list[ActionItem]
    """
    List of ActionItem objects with tasks, owners, and deadlines.
    Example: {task: "Update API docs", owner: "Bob", deadline: "Friday"}
    """
    
    # =========================================================================
    # Error Handling
    # =========================================================================
    # Nodes can set this field to report errors without crashing.
    
    error: Optional[str]
    """
    Error message if any node encountered a problem.
    When set, downstream nodes can check this and handle gracefully.
    """
