"""
Pydantic Data Models (Schemas)

This module defines all data structures used throughout the application.
Pydantic models provide validation, serialization, and documentation.

=============================================================================
WHY PYDANTIC?
=============================================================================

1. VALIDATION: Automatic input validation with helpful error messages
2. SERIALIZATION: Easy conversion to/from JSON, dict
3. TYPE SAFETY: Full IDE support with type hints
4. DOCUMENTATION: Auto-generated OpenAPI schemas for API docs

=============================================================================
MODEL HIERARCHY:
=============================================================================

    TranscriptSegment     ← Individual line from transcript
           │
           ▼
    Meeting               ← Complete meeting with transcript
           │
           ├── segments[] ← List of TranscriptSegment
           ├── decisions[] ← List of Decision (from analysis)
           └── action_items[] ← List of ActionItem (from analysis)

    Decision              ← Key decision made in meeting
    ActionItem            ← Task assigned to someone

=============================================================================
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """
    A single segment of meeting transcript.
    
    Represents one spoken statement in the meeting, typically
    corresponding to one line in the transcript.
    
    EXAMPLE:
        Input line: "[00:05:30] Alice: I think we should extend the deadline"
        
        TranscriptSegment(
            speaker="Alice",
            timestamp="00:05:30",
            text="I think we should extend the deadline"
        )
    
    ATTRIBUTES:
        speaker: Name of the person speaking (extracted from transcript)
        timestamp: When this was said (format varies: "00:00" or "00:00:00")
        text: What was actually said
    """
    
    speaker: str = Field(
        description="Name of the speaker"
    )
    timestamp: str = Field(
        default="00:00",
        description="Timestamp in format HH:MM or HH:MM:SS"
    )
    text: str = Field(
        description="The spoken content"
    )


class Decision(BaseModel):
    """
    A key decision made during a meeting.
    
    Extracted by the decisions_node in the LangGraph pipeline.
    Represents an agreement, conclusion, or direction change.
    
    EXAMPLE:
        Decision(
            decision="Extend project deadline by 2 weeks",
            made_by="Project Manager Alice",
            context="Team needs more time for QA testing",
            related_discussion="Sprint capacity and testing requirements"
        )
    
    WHAT COUNTS AS A DECISION:
        ✓ "We decided to use Python"
        ✓ "The team agreed on a Friday deadline"
        ✓ "We're pivoting from approach A to B"
        ✗ "Maybe we should consider..." (just a suggestion)
        ✗ "What if we tried..." (not final)
    """
    
    decision: str = Field(
        description="The specific decision or agreement made"
    )
    made_by: Optional[str] = Field(
        default=None,
        description="Person who made or announced the decision"
    )
    context: Optional[str] = Field(
        default=None,
        description="Reasoning or background behind the decision"
    )
    related_discussion: Optional[str] = Field(
        default=None,
        description="Key points from the discussion leading to this decision"
    )


class ActionItem(BaseModel):
    """
    A task or follow-up action from the meeting.
    
    Extracted by the actions_node in the LangGraph pipeline.
    Represents something that needs to be done after the meeting.
    
    EXAMPLE:
        ActionItem(
            task="Update the API documentation",
            owner="Bob",
            deadline="End of week",
            priority="high",
            context="Needed before the client demo"
        )
    
    PRIORITY INTERPRETATION:
        high   → Use "ASAP", "urgent", "critical", "by EOD"
        medium → Use "this week", "soon", "when possible"
        low    → Use "when you have time", "nice to have"
    """
    
    task: str = Field(
        description="The specific task or action to be done"
    )
    owner: Optional[str] = Field(
        default=None,
        description="Person responsible for completing the task"
    )
    deadline: Optional[str] = Field(
        default=None,
        description="When the task should be completed (if mentioned)"
    )
    priority: Optional[str] = Field(
        default=None,
        description="Priority level: high, medium, or low"
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context about why this task matters"
    )


class Meeting(BaseModel):
    """
    Complete meeting record with transcript and analysis.
    
    This is the main data structure for a meeting. It includes:
    - Basic metadata (id, title, timestamps)
    - The raw and parsed transcript
    - Analysis results (summary, decisions, action items)
    
    LIFECYCLE:
    ----------
    
    1. UPLOAD:
       Meeting is created with id, title, raw_transcript, segments
       Analysis fields (summary, decisions, etc.) are None
    
    2. ANALYZE:
       LangGraph pipeline runs and populates:
       - summary
       - key_topics
       - decisions
       - action_items
    
    3. QUERY:
       Q&A agent uses vector store to answer questions about this meeting
    
    EXAMPLE:
        meeting = Meeting(
            id="abc-123",
            title="Sprint Planning",
            raw_transcript="[00:00] Alice: Welcome everyone...",
            segments=[TranscriptSegment(...)],
            participants=["Alice", "Bob", "Carol"]
        )
    """
    
    # =========================================================================
    # Basic Metadata
    # =========================================================================
    
    id: str = Field(
        description="Unique identifier (UUID format)"
    )
    
    title: str = Field(
        description="Human-readable meeting title"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the meeting was uploaded"
    )
    
    # =========================================================================
    # Transcript Data
    # =========================================================================
    
    raw_transcript: Optional[str] = Field(
        default=None,
        description="Original unprocessed transcript text"
    )
    
    segments: list[TranscriptSegment] = Field(
        default_factory=list,
        description="Parsed transcript segments with speaker/timestamp"
    )
    
    participants: list[str] = Field(
        default_factory=list,
        description="List of unique speaker names"
    )
    
    # =========================================================================
    # Analysis Results (populated by LangGraph pipeline)
    # =========================================================================
    
    summary: Optional[str] = Field(
        default=None,
        description="LLM-generated meeting summary"
    )
    
    key_topics: list[str] = Field(
        default_factory=list,
        description="Main topics discussed in the meeting"
    )
    
    decisions: list[Decision] = Field(
        default_factory=list,
        description="Key decisions extracted from the meeting"
    )
    
    action_items: list[ActionItem] = Field(
        default_factory=list,
        description="Action items with owners and deadlines"
    )


# =============================================================================
# RESPONSE MODELS (for API responses)
# =============================================================================
# These are used by FastAPI to generate automatic API documentation
# and validate response formats.

class AnalysisResult(BaseModel):
    """
    Response model for meeting analysis endpoint.
    
    Returned by POST /api/v1/meetings/{id}/analyze
    """
    
    meeting_id: str = Field(
        description="ID of the analyzed meeting"
    )
    
    summary: str = Field(
        description="Generated meeting summary"
    )
    
    key_topics: list[str] = Field(
        description="Main topics discussed"
    )
    
    decisions: list[Decision] = Field(
        description="Decisions extracted from meeting"
    )
    
    action_items: list[ActionItem] = Field(
        description="Action items extracted from meeting"
    )


class QuestionAnswer(BaseModel):
    """
    Response model for Q&A endpoint.
    
    Returned by POST /api/v1/meetings/{id}/ask
    """
    
    answer: str = Field(
        description="Generated answer to the question"
    )
    
    sources: list[str] = Field(
        default_factory=list,
        description="Transcript excerpts used to generate the answer"
    )
