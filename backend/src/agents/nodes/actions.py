"""
Action Item Agent Node - Extract Tasks and Assignments

This node identifies action items, tasks, and assignments
from the meeting transcript with owners and deadlines.
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ...models import ActionItem
from ...services import get_llm_service
from ..state import AgentState

logger = logging.getLogger(__name__)

ACTION_ITEM_SYSTEM_PROMPT = """You are an expert at analyzing meeting transcripts to identify 
action items, tasks, and assignments.

Your task is to extract:
1. Tasks that were assigned to specific people
2. Follow-up actions mentioned
3. Commitments made by participants
4. Deadlines or timelines mentioned

For each action item, identify:
- The specific task or action (what needs to be done)
- The owner (who is responsible - use exact names from transcript)
- Any deadline mentioned
- Priority (high/medium/low based on urgency/importance)
- Context from the discussion

Output your findings as a JSON array with this structure:
[
    {
        "task": "Description of the task",
        "owner": "Person responsible or null",
        "deadline": "Deadline if mentioned or null",
        "priority": "high/medium/low or null",
        "context": "Brief context from discussion"
    }
]

Guidelines:
- Be specific about what needs to be done
- Use exact participant names as owners
- Only include items that are clearly tasks, not suggestions
- Infer priority from language (ASAP = high, "when you can" = low)
- If no clear action items, return an empty array []
- Maximum 15 action items per meeting"""

ACTION_ITEM_USER_PROMPT = """Analyze the following meeting transcript and extract all action items:

Meeting Title: {title}
Participants: {participants}

--- TRANSCRIPT ---
{transcript}
--- END TRANSCRIPT ---

Extract the action items as a JSON array."""


async def actions_node(state: AgentState) -> dict:
    """
    LangGraph node that extracts action items from meetings.
    
    This node:
    1. Analyzes the transcript for task assignments
    2. Identifies owners and deadlines
    3. Assigns priority based on context
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state with action_items list
    """
    logger.info(f"Extracting action items for meeting: {state['meeting_id']}")
    
    try:
        llm_service = get_llm_service()
        
        # Use parsed transcript if available
        transcript = state.get("parsed_transcript") or state["raw_transcript"]
        participants = ", ".join(state.get("participants", [])) or "Not specified"
        
        # Create the prompt
        user_message = ACTION_ITEM_USER_PROMPT.format(
            title=state["meeting_title"],
            participants=participants,
            transcript=transcript[:15000],
        )
        
        messages = [
            SystemMessage(content=ACTION_ITEM_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        
        # Generate action item extraction
        response = await llm_service.chat(messages)
        
        # Parse the JSON response
        action_items = _parse_action_items(response)
        
        logger.info(f"Extracted {len(action_items)} action items")
        
        return {
            "action_items": action_items,
        }
    
    except Exception as e:
        logger.error(f"Error extracting action items: {e}")
        return {
            "error": f"Failed to extract action items: {str(e)}",
            "action_items": [],
        }


def _parse_action_items(response: str) -> list[ActionItem]:
    """
    Parse the LLM response into ActionItem objects.
    
    Args:
        response: LLM response containing JSON
    
    Returns:
        List of ActionItem objects
    """
    action_items = []
    
    try:
        # Find JSON array in response
        json_str = _extract_json_array(response)
        
        if json_str:
            data = json.loads(json_str)
            
            for item in data:
                if isinstance(item, dict) and "task" in item:
                    action_items.append(
                        ActionItem(
                            task=item.get("task", ""),
                            owner=item.get("owner"),
                            deadline=item.get("deadline"),
                            priority=item.get("priority"),
                            context=item.get("context"),
                        )
                    )
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse action items JSON: {e}")
    except Exception as e:
        logger.error(f"Error parsing action items: {e}")
    
    return action_items


def _extract_json_array(text: str) -> str | None:
    """
    Extract JSON array from text that may contain other content.
    
    Args:
        text: Text containing JSON array
    
    Returns:
        The JSON array string, or None if not found
    """
    start = text.find("[")
    end = text.rfind("]")
    
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    
    return None
