"""
FastAPI Routes for Meeting Intelligence API

This module defines the REST API endpoints for the meeting
intelligence system.
"""

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..agents import get_meeting_analyzer, get_qa_agent
from ..agents.nodes import parse_transcript
from ..models import (
    AnalysisRequest,
    AnalysisResponse,
    AudioTranscriptionResponse,
    ChatRequest,
    ChatResponse,
    Meeting,
    MeetingSummary,
    TranscriptUploadResponse,
)
from ..services import get_whisper_service
from ..vectorstore import get_chroma_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["meetings"])


# Request/Response models for API
class TranscriptTextRequest(BaseModel):
    """Request to upload transcript as text."""
    title: str
    transcript: str


class QuestionRequest(BaseModel):
    """Request to ask a question about a meeting."""
    meeting_id: str
    question: str


class QuestionResponse(BaseModel):
    """Response to a question."""
    answer: str
    sources: list[str]


# In-memory meeting storage (in production, use a database)
_meetings: dict[str, dict] = {}


@router.post("/transcripts/upload", response_model=TranscriptUploadResponse)
async def upload_transcript(request: TranscriptTextRequest):
    """
    Upload a meeting transcript as text.
    
    The transcript will be parsed, chunked, and stored in the vector database
    for later querying.
    """
    try:
        meeting_id = str(uuid.uuid4())
        
        # Parse the transcript
        segments = parse_transcript(request.transcript)
        participants = list({seg.speaker for seg in segments if seg.speaker != "Unknown"})
        
        # Store in vector database
        vector_store = get_chroma_store()
        
        meeting = Meeting(
            id=meeting_id,
            title=request.title,
            segments=segments,
            participants=participants,
            raw_transcript=request.transcript,
        )
        
        chunk_count = vector_store.add_meeting(meeting)
        
        # Store meeting metadata
        _meetings[meeting_id] = {
            "id": meeting_id,
            "title": request.title,
            "participants": participants,
            "segment_count": len(segments),
            "raw_transcript": request.transcript,
        }
        
        logger.info(f"Uploaded transcript: {meeting_id} with {chunk_count} chunks")
        
        return TranscriptUploadResponse(
            meeting_id=meeting_id,
            title=request.title,
            segment_count=len(segments),
            participant_count=len(participants),
            message=f"Transcript uploaded successfully with {chunk_count} chunks indexed",
        )
    
    except Exception as e:
        logger.error(f"Error uploading transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcripts/upload-file", response_model=TranscriptUploadResponse)
async def upload_transcript_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
):
    """
    Upload a meeting transcript from a text file.
    """
    try:
        # Read file content
        content = await file.read()
        transcript_text = content.decode("utf-8")
        
        # Use filename as title if not provided
        file_title = title or file.filename or "Untitled Meeting"
        
        # Delegate to the text upload endpoint
        request = TranscriptTextRequest(title=file_title, transcript=transcript_text)
        return await upload_transcript(request)
    
    except Exception as e:
        logger.error(f"Error uploading transcript file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audio/transcribe", response_model=AudioTranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
):
    """
    Transcribe an audio file using Whisper (local).
    
    Supported formats: MP3, WAV, M4A, OGG, FLAC
    """
    try:
        # Get file extension
        filename = file.filename or "audio.wav"
        extension = "." + filename.split(".")[-1].lower()
        
        # Read audio bytes
        audio_bytes = await file.read()
        
        # Transcribe using Whisper
        whisper_service = get_whisper_service()
        segments, detected_language, duration = whisper_service.transcribe_bytes(
            audio_bytes=audio_bytes,
            file_extension=extension,
            language=language,
        )
        
        # Generate meeting ID
        meeting_id = str(uuid.uuid4())
        file_title = title or filename.rsplit(".", 1)[0]
        
        # Build full transcript text
        transcript_text = "\n".join(
            f"[{seg.timestamp}] {seg.speaker}: {seg.text}"
            for seg in segments
        )
        
        # Store in vector database
        vector_store = get_chroma_store()
        meeting = Meeting(
            id=meeting_id,
            title=file_title,
            segments=segments,
            participants=["Speaker"],  # Whisper doesn't do diarization
            raw_transcript=transcript_text,
        )
        vector_store.add_meeting(meeting)
        
        # Store meeting metadata
        _meetings[meeting_id] = {
            "id": meeting_id,
            "title": file_title,
            "participants": ["Speaker"],
            "segment_count": len(segments),
            "raw_transcript": transcript_text,
        }
        
        logger.info(f"Transcribed audio: {meeting_id}, duration: {duration}s")
        
        return AudioTranscriptionResponse(
            meeting_id=meeting_id,
            transcript=transcript_text,
            segments=segments,
            language=detected_language,
            duration_seconds=duration,
        )
    
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/meetings/{meeting_id}/analyze", response_model=AnalysisResponse)
async def analyze_meeting(meeting_id: str):
    """
    Run full analysis on a meeting transcript.
    
    Generates summary, extracts decisions, and identifies action items.
    """
    if meeting_id not in _meetings:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    try:
        start_time = time.time()
        
        meeting_data = _meetings[meeting_id]
        analyzer = get_meeting_analyzer()
        
        result = await analyzer.analyze(
            meeting_id=meeting_id,
            meeting_title=meeting_data["title"],
            raw_transcript=meeting_data["raw_transcript"],
        )
        
        processing_time = time.time() - start_time
        
        # Create summary object
        summary = MeetingSummary(
            overview=result.get("summary", ""),
            key_topics=result.get("key_topics", []),
            decisions=result.get("decisions", []),
            action_items=result.get("action_items", []),
            participants_summary={},
        )
        
        # Update stored meeting with analysis results
        _meetings[meeting_id]["analysis"] = result
        
        return AnalysisResponse(
            meeting_id=meeting_id,
            summary=summary,
            processing_time_seconds=processing_time,
        )
    
    except Exception as e:
        logger.error(f"Error analyzing meeting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/meetings/{meeting_id}/ask", response_model=QuestionResponse)
async def ask_question(meeting_id: str, request: QuestionRequest):
    """
    Ask a question about a meeting transcript.
    """
    if meeting_id not in _meetings:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    try:
        qa_agent = get_qa_agent()
        
        result = await qa_agent.ask(
            question=request.question,
            meeting_id=meeting_id,
        )
        
        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"],
        )
    
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/meetings")
async def list_meetings():
    """
    List all uploaded meetings.
    """
    return [
        {
            "id": m["id"],
            "title": m["title"],
            "participants": m.get("participants", []),
            "segment_count": m.get("segment_count", 0),
            "has_analysis": "analysis" in m,
        }
        for m in _meetings.values()
    ]


@router.get("/meetings/{meeting_id}")
async def get_meeting(meeting_id: str):
    """
    Get details about a specific meeting.
    """
    if meeting_id not in _meetings:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    return _meetings[meeting_id]


@router.delete("/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str):
    """
    Delete a meeting and its vector store data.
    """
    if meeting_id not in _meetings:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    try:
        # Remove from vector store
        vector_store = get_chroma_store()
        vector_store.delete_meeting(meeting_id)
        
        # Remove from memory
        del _meetings[meeting_id]
        
        return {"message": f"Meeting {meeting_id} deleted successfully"}
    
    except Exception as e:
        logger.error(f"Error deleting meeting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "service": "meeting-intelligence-api"}
