"""
API Routes - FastAPI Endpoint Definitions

This module defines all REST API endpoints for the Meeting Intelligence System.

=============================================================================
API STRUCTURE:
=============================================================================

All endpoints are prefixed with /api/v1 (see main.py)

TRANSCRIPT ENDPOINTS:
    POST /transcripts/upload      - Upload text transcript
    POST /transcripts/upload-file - Upload transcript file (.txt)

AUDIO ENDPOINTS:
    POST /audio/transcribe        - Upload audio and convert to transcript

MEETING ENDPOINTS:
    GET  /meetings                - List all meetings
    GET  /meetings/{id}           - Get meeting details
    DELETE /meetings/{id}         - Delete a meeting
    POST /meetings/{id}/analyze   - Run AI analysis on a meeting
    POST /meetings/{id}/ask       - Ask a question (Q&A)

HEALTH ENDPOINT:
    GET /health                   - Service health check

=============================================================================
DATA FLOW EXAMPLE (Transcript Upload):
=============================================================================

    1. Client POSTs transcript text to /transcripts/upload
    2. Backend parses transcript into segments
    3. Segments are stored in ChromaDB (vector store)
    4. Meeting is added to in-memory meeting list
    5. Response includes meeting_id for future operations

=============================================================================
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

# Import our internal modules
from ..agents import MeetingAnalyzer, get_qa_agent
from ..agents.nodes import parse_transcript
from ..models import Meeting, TranscriptSegment
from ..services import get_whisper_service
from ..vectorstore import get_chroma_store

logger = logging.getLogger(__name__)

# =============================================================================
# API ROUTER
# =============================================================================
# APIRouter allows us to define routes separately from the main app.
# This enables modular organization (routes in separate file from main.py)
router = APIRouter()


# =============================================================================
# IN-MEMORY MEETING STORAGE
# =============================================================================
# For this demo/prototype, we store meetings in memory.
# PRODUCTION NOTE: Replace with a proper database (PostgreSQL, etc.)
#
# Structure: {meeting_id: Meeting object}
meetings_db: dict[str, Meeting] = {}


# =============================================================================
# REQUEST/RESPONSE MODELS (Pydantic)
# =============================================================================
# Pydantic models provide:
# 1. Request validation (automatic 422 errors for invalid data)
# 2. Response serialization (automatic JSON conversion)
# 3. API documentation (auto-generated OpenAPI/Swagger docs)

class TranscriptUploadRequest(BaseModel):
    """Request body for uploading a text transcript."""
    title: str              # Meeting title (required)
    transcript: str         # Full transcript text (required)


class QuestionRequest(BaseModel):
    """Request body for asking questions about a meeting."""
    meeting_id: str         # Which meeting to query
    question: str           # The question to ask


# =============================================================================
# TRANSCRIPT ENDPOINTS
# =============================================================================

@router.post("/transcripts/upload")
async def upload_transcript(request: TranscriptUploadRequest):
    """
    Upload a text transcript for analysis.
    
    This endpoint:
    1. Parses the transcript into structured segments
    2. Stores segments in the vector database for semantic search
    3. Saves meeting metadata for later retrieval
    
    REQUEST BODY:
        {
            "title": "Sprint Planning Meeting",
            "transcript": "[00:00] Alice: Welcome everyone..."
        }
    
    RESPONSE:
        {
            "meeting_id": "abc123-def456",
            "segment_count": 42,
            "participants": ["Alice", "Bob", "Carol"]
        }
    
    TRANSCRIPT FORMAT:
        The parser supports multiple formats:
        - [00:00] Speaker: Text
        - 00:00 - Speaker: Text
        - Speaker (00:00): Text
        - Speaker: Text (no timestamp)
    """
    try:
        # Generate unique ID for this meeting
        meeting_id = str(uuid.uuid4())
        
        # Parse transcript into structured segments
        # This extracts: timestamp, speaker, text for each line
        segments = parse_transcript(request.transcript)
        
        # Extract unique participant names from segments
        participants = list({seg.speaker for seg in segments if seg.speaker})
        
        # Create Meeting object to store
        meeting = Meeting(
            id=meeting_id,
            title=request.title,
            raw_transcript=request.transcript,
            segments=segments,
            participants=participants,
        )
        
        # Store in our in-memory "database"
        meetings_db[meeting_id] = meeting
        
        # Add to vector store for semantic search
        # This embeds each segment and stores in ChromaDB
        vector_store = get_chroma_store()
        vector_store.add_meeting(meeting)
        
        logger.info(
            f"Uploaded meeting '{request.title}' with {len(segments)} segments"
        )
        
        return {
            "meeting_id": meeting_id,
            "segment_count": len(segments),
            "participants": participants,
        }
        
    except Exception as e:
        logger.error(f"Failed to upload transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcripts/upload-file")
async def upload_transcript_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
):
    """
    Upload a transcript file (.txt format).
    
    Similar to /transcripts/upload but accepts a file upload.
    If title is not provided, uses the filename.
    
    REQUEST:
        multipart/form-data with:
        - file: .txt file containing the transcript
        - title: (optional) Meeting title
    """
    try:
        # Read file content
        content = await file.read()
        transcript_text = content.decode("utf-8")
        
        # Use filename as title if not provided
        meeting_title = title or file.filename.replace(".txt", "")
        
        # Reuse the upload logic
        request = TranscriptUploadRequest(
            title=meeting_title,
            transcript=transcript_text,
        )
        
        return await upload_transcript(request)
        
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# AUDIO TRANSCRIPTION ENDPOINT
# =============================================================================

@router.post("/audio/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
):
    """
    Upload audio file and transcribe using local Whisper.
    
    SUPPORTED FORMATS:
        MP3, WAV, M4A, OGG, FLAC
    
    HOW IT WORKS:
        1. Audio file is saved temporarily
        2. Whisper (faster-whisper) processes it locally
        3. Transcription is returned with timestamps
        4. Meeting is stored like a regular transcript upload
    
    IMPORTANT:
        - Whisper runs LOCALLY (no API calls, no cost)
        - First call downloads the model (~74MB for 'base')
        - Large files may take several minutes
    
    REQUEST:
        multipart/form-data with:
        - file: Audio file (mp3, wav, etc.)
        - title: (optional) Meeting title
        - language: (optional) Language code (en, es, fr, etc.)
    """
    try:
        # Get the Whisper service (singleton - model loaded once)
        whisper_service = get_whisper_service()
        
        # Read audio file bytes
        audio_bytes = await file.read()
        
        # Determine file extension from filename
        file_ext = f".{file.filename.split('.')[-1]}" if file.filename else ".wav"
        
        # Transcribe using Whisper
        # Returns: (segments, detected_language, duration_seconds)
        segments, detected_lang, duration = whisper_service.transcribe_bytes(
            audio_bytes=audio_bytes,
            file_extension=file_ext,
            language=language,
        )
        
        # Generate meeting ID and title
        meeting_id = str(uuid.uuid4())
        meeting_title = title or file.filename or "Audio Transcription"
        
        # Build raw transcript from segments
        # Format: [timestamp] Speaker: text
        raw_transcript = "\n".join([
            f"[{seg.timestamp}] {seg.speaker}: {seg.text}"
            for seg in segments
        ])
        
        # Extract participants (Whisper doesn't do speaker diarization,
        # so all segments have generic "Speaker" label)
        participants = list({seg.speaker for seg in segments})
        
        # Create and store meeting
        meeting = Meeting(
            id=meeting_id,
            title=meeting_title,
            raw_transcript=raw_transcript,
            segments=segments,
            participants=participants,
        )
        
        meetings_db[meeting_id] = meeting
        
        # Add to vector store
        vector_store = get_chroma_store()
        vector_store.add_meeting(meeting)
        
        logger.info(
            f"Transcribed audio '{meeting_title}': "
            f"{len(segments)} segments, {duration:.1f}s, language={detected_lang}"
        )
        
        return {
            "meeting_id": meeting_id,
            "segments": [seg.model_dump() for seg in segments],
            "language": detected_lang,
            "duration_seconds": duration,
        }
        
    except Exception as e:
        logger.error(f"Failed to transcribe audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MEETING MANAGEMENT ENDPOINTS
# =============================================================================

@router.get("/meetings")
async def list_meetings():
    """
    List all uploaded meetings.
    
    Returns basic metadata for each meeting (not full transcripts).
    Useful for populating a meeting list in the UI.
    """
    return [
        {
            "id": m.id,
            "title": m.title,
            "participants": m.participants,
            "has_analysis": m.summary is not None,  # Has been analyzed?
        }
        for m in meetings_db.values()
    ]


@router.get("/meetings/{meeting_id}")
async def get_meeting(meeting_id: str):
    """
    Get full details for a specific meeting.
    
    Includes the complete transcript and any analysis results.
    """
    meeting = meetings_db.get(meeting_id)
    
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    return meeting.model_dump()


@router.delete("/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str):
    """
    Delete a meeting and its vector store data.
    
    This removes:
    1. The meeting from our in-memory database
    2. All embedded chunks from ChromaDB
    """
    if meeting_id not in meetings_db:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    # Remove from vector store first
    vector_store = get_chroma_store()
    vector_store.delete_meeting(meeting_id)
    
    # Remove from our database
    del meetings_db[meeting_id]
    
    logger.info(f"Deleted meeting: {meeting_id}")
    
    return {"status": "deleted", "meeting_id": meeting_id}


# =============================================================================
# AI ANALYSIS ENDPOINTS
# =============================================================================

@router.post("/meetings/{meeting_id}/analyze")
async def analyze_meeting(meeting_id: str):
    """
    Run AI analysis on a meeting transcript.
    
    This processes the transcript through the LangGraph pipeline:
    1. Parser: Normalize format, extract speakers
    2. Summarizer: Generate high-level overview (uses LLM)
    3. Decisions: Extract key decisions (uses LLM)
    4. Actions: Extract action items (uses LLM)
    
    IMPORTANT:
        - This makes multiple LLM API calls (costs $)
        - May take 10-30 seconds depending on transcript length
        - Results are cached in the meeting object
    
    RESPONSE:
        {
            "meeting_id": "abc123",
            "summary": {
                "overview": "This meeting discussed...",
                "key_topics": ["Budget", "Timeline", "Resources"],
                "decisions": [...],
                "action_items": [...]
            }
        }
    """
    meeting = meetings_db.get(meeting_id)
    
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    try:
        # Create analyzer and run the LangGraph pipeline
        analyzer = MeetingAnalyzer()
        result = await analyzer.analyze(
            meeting_id=meeting_id,
            meeting_title=meeting.title,
            raw_transcript=meeting.raw_transcript,
        )
        
        # Update meeting with analysis results
        # (Cache so we don't re-analyze on every request)
        meeting.summary = result.get("summary")
        meeting.key_topics = result.get("key_topics", [])
        meeting.decisions = result.get("decisions", [])
        meeting.action_items = result.get("action_items", [])
        
        return {
            "meeting_id": meeting_id,
            "summary": {
                "overview": meeting.summary,
                "key_topics": meeting.key_topics,
                "decisions": [d.model_dump() for d in meeting.decisions],
                "action_items": [a.model_dump() for a in meeting.action_items],
            },
        }
        
    except Exception as e:
        logger.error(f"Analysis failed for {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/meetings/{meeting_id}/ask")
async def ask_question(meeting_id: str, request: QuestionRequest):
    """
    Ask a question about a meeting using RAG.
    
    This uses the Q&A Agent which:
    1. Searches the vector store for relevant transcript chunks
    2. Passes the context to an LLM
    3. Returns an answer grounded in the meeting content
    
    REQUEST:
        {
            "meeting_id": "abc123",
            "question": "What was decided about the deadline?"
        }
    
    RESPONSE:
        {
            "answer": "The team decided to extend the deadline...",
            "sources": ["[00:15] Alice: I think we need more time..."]
        }
    """
    if meeting_id not in meetings_db:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    try:
        # Get Q&A agent and ask the question
        qa_agent = get_qa_agent()
        result = await qa_agent.ask(
            question=request.question,
            meeting_id=meeting_id,
        )
        
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
        }
        
    except Exception as e:
        logger.error(f"Q&A failed for {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HEALTH CHECK ENDPOINT
# =============================================================================

@router.get("/health")
async def health_check():
    """
    Service health check endpoint.
    
    Used by:
    - Docker health checks
    - Kubernetes liveness/readiness probes
    - Monitoring systems
    
    Returns 200 if the service is running.
    """
    return {
        "status": "healthy",
        "service": "meeting-intelligence-api",
    }
