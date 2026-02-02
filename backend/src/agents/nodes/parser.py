"""
Parser Node - Transcript Normalization

This node parses and normalizes meeting transcripts,
extracting speaker labels, timestamps, and structured text.
"""

import logging
import re
from typing import Any

from ...models import TranscriptSegment
from ..state import AgentState

logger = logging.getLogger(__name__)


# Common transcript patterns
PATTERNS = {
    # [00:05:23] Speaker Name: Text
    "bracket_timestamp": re.compile(
        r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*([^:]+):\s*(.+)"
    ),
    # 00:05:23 - Speaker Name: Text
    "dash_timestamp": re.compile(
        r"(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–]\s*([^:]+):\s*(.+)"
    ),
    # Speaker Name (00:05:23): Text
    "paren_timestamp": re.compile(
        r"([^(]+)\s*\((\d{1,2}:\d{2}(?::\d{2})?)\):\s*(.+)"
    ),
    # Speaker Name: Text (no timestamp)
    "simple_speaker": re.compile(
        r"^([A-Z][a-zA-Z\s]+):\s*(.+)"
    ),
}


def parse_transcript_line(line: str) -> dict[str, Any] | None:
    """
    Parse a single line of transcript.
    
    Args:
        line: A line from the transcript
    
    Returns:
        Dictionary with speaker, timestamp, and text, or None if not parseable
    """
    line = line.strip()
    if not line:
        return None
    
    # Try each pattern
    for pattern_name, pattern in PATTERNS.items():
        match = pattern.match(line)
        if match:
            groups = match.groups()
            
            if pattern_name == "bracket_timestamp":
                return {
                    "timestamp": groups[0],
                    "speaker": groups[1].strip(),
                    "text": groups[2].strip(),
                }
            elif pattern_name == "dash_timestamp":
                return {
                    "timestamp": groups[0],
                    "speaker": groups[1].strip(),
                    "text": groups[2].strip(),
                }
            elif pattern_name == "paren_timestamp":
                return {
                    "timestamp": groups[1],
                    "speaker": groups[0].strip(),
                    "text": groups[2].strip(),
                }
            elif pattern_name == "simple_speaker":
                return {
                    "timestamp": "00:00",
                    "speaker": groups[0].strip(),
                    "text": groups[1].strip(),
                }
    
    return None


def parse_transcript(raw_transcript: str) -> list[TranscriptSegment]:
    """
    Parse a full transcript into segments.
    
    Args:
        raw_transcript: The raw transcript text
    
    Returns:
        List of TranscriptSegment objects
    """
    segments: list[TranscriptSegment] = []
    current_speaker = "Unknown"
    current_timestamp = "00:00"
    
    for line in raw_transcript.split("\n"):
        parsed = parse_transcript_line(line)
        
        if parsed:
            segments.append(
                TranscriptSegment(
                    speaker=parsed["speaker"],
                    timestamp=parsed["timestamp"],
                    text=parsed["text"],
                )
            )
            current_speaker = parsed["speaker"]
            current_timestamp = parsed["timestamp"]
        elif line.strip():
            # Continuation of previous speaker's text
            if segments:
                # Append to previous segment
                segments[-1].text += " " + line.strip()
            else:
                # First line without format - treat as content
                segments.append(
                    TranscriptSegment(
                        speaker=current_speaker,
                        timestamp=current_timestamp,
                        text=line.strip(),
                    )
                )
    
    return segments


def format_parsed_transcript(segments: list[TranscriptSegment]) -> str:
    """
    Format parsed segments into a normalized transcript.
    
    Args:
        segments: List of TranscriptSegment objects
    
    Returns:
        Formatted transcript string
    """
    lines = []
    for seg in segments:
        lines.append(f"[{seg.timestamp}] {seg.speaker}: {seg.text}")
    return "\n".join(lines)


def extract_participants(segments: list[TranscriptSegment]) -> list[str]:
    """
    Extract unique participant names from segments.
    
    Args:
        segments: List of TranscriptSegment objects
    
    Returns:
        List of unique speaker names
    """
    speakers = set()
    for seg in segments:
        if seg.speaker and seg.speaker != "Unknown":
            speakers.add(seg.speaker)
    return sorted(list(speakers))


async def parser_node(state: AgentState) -> dict:
    """
    LangGraph node that parses and normalizes transcripts.
    
    This node:
    1. Parses the raw transcript into structured segments
    2. Extracts participant names
    3. Creates a normalized transcript format
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state fields
    """
    logger.info(f"Parsing transcript for meeting: {state['meeting_id']}")
    
    try:
        raw_transcript = state["raw_transcript"]
        
        # Parse the transcript
        segments = parse_transcript(raw_transcript)
        
        # Extract participants
        participants = extract_participants(segments)
        
        # Format normalized transcript
        parsed_transcript = format_parsed_transcript(segments)
        
        logger.info(
            f"Parsed {len(segments)} segments, "
            f"found {len(participants)} participants"
        )
        
        return {
            "parsed_transcript": parsed_transcript,
            "participants": participants,
        }
    
    except Exception as e:
        logger.error(f"Error parsing transcript: {e}")
        return {
            "error": f"Failed to parse transcript: {str(e)}",
            "parsed_transcript": state["raw_transcript"],
            "participants": [],
        }
