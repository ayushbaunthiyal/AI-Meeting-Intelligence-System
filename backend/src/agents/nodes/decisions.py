"""
Decision Extractor Node - Identify Key Decisions

This node identifies and extracts key decisions, agreements,
and pivotal moments from the meeting transcript.
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...models import Decision
from ...services import get_llm_service
from ..state import AgentState

logger = logging.getLogger(__name__)

DECISION_EXTRACTOR_SYSTEM_PROMPT = """You are an expert at analyzing meeting transcripts to identify 
key decisions and agreements made during discussions.

Your task is to extract:
1. Final decisions that were made
2. Agreements reached between participants
3. Key pivots or direction changes
4. Approvals or rejections of proposals

For each decision, identify:
- The decision itself (what was decided)
- Who made or announced the decision (if mentioned)
- The context or reasoning behind it
- Related discussion points

Output your findings as a JSON array with this structure:
[
    {
        "decision": "Description of the decision",
        "made_by": "Person who made/announced it or null",
        "context": "Brief context or reasoning",
        "related_discussion": "Related discussion points"
    }
]

Guidelines:
- Only include actual decisions, not proposals or suggestions
- Be specific and factual
- If no clear decisions were made, return an empty array []
- Maximum 10 decisions per meeting"""

DECISION_EXTRACTOR_USER_PROMPT = """Analyze the following meeting transcript and extract all key decisions made:

Meeting Title: {title}
Participants: {participants}

--- TRANSCRIPT ---
{transcript}
--- END TRANSCRIPT ---

Extract the decisions as a JSON array."""


async def decisions_node(state: AgentState) -> dict:
    """
    LangGraph node that extracts decisions from meetings.
    
    This node:
    1. Analyzes the transcript for decision points
    2. Extracts structured decision information
    3. Links decisions to participants who made them
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state with decisions list
    """
    logger.info(f"Extracting decisions for meeting: {state['meeting_id']}")
    
    try:
        llm_service = get_llm_service()
        
        # Use parsed transcript if available
        transcript = state.get("parsed_transcript") or state["raw_transcript"]
        participants = ", ".join(state.get("participants", [])) or "Not specified"
        
        # Create the prompt
        user_message = DECISION_EXTRACTOR_USER_PROMPT.format(
            title=state["meeting_title"],
            participants=participants,
            transcript=transcript[:15000],
        )
        
        messages = [
            SystemMessage(content=DECISION_EXTRACTOR_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        
        # Generate decision extraction
        response = await llm_service.chat(messages)
        
        # Parse the JSON response
        decisions = _parse_decisions(response)
        
        logger.info(f"Extracted {len(decisions)} decisions")
        
        return {
            "decisions": decisions,
        }
    
    except Exception as e:
        logger.error(f"Error extracting decisions: {e}")
        return {
            "error": f"Failed to extract decisions: {str(e)}",
            "decisions": [],
        }


def _parse_decisions(response: str) -> list[Decision]:
    """
    Parse the LLM response into Decision objects.
    
    Args:
        response: LLM response containing JSON
    
    Returns:
        List of Decision objects
    """
    decisions = []
    
    try:
        # Find JSON array in response
        json_str = _extract_json_array(response)
        
        if json_str:
            data = json.loads(json_str)
            
            for item in data:
                if isinstance(item, dict) and "decision" in item:
                    decisions.append(
                        Decision(
                            decision=item.get("decision", ""),
                            made_by=item.get("made_by"),
                            context=item.get("context"),
                            related_discussion=item.get("related_discussion"),
                        )
                    )
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse decisions JSON: {e}")
    except Exception as e:
        logger.error(f"Error parsing decisions: {e}")
    
    return decisions


def _extract_json_array(text: str) -> str | None:
    """
    Extract JSON array from text that may contain other content.
    
    Args:
        text: Text containing JSON array
    
    Returns:
        The JSON array string, or None if not found
    """
    # Find the first [ and last ]
    start = text.find("[")
    end = text.rfind("]")
    
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    
    return None
