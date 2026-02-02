"""
Summarizer Node - Meeting Summary Generation

This node generates a high-level overview of the meeting
discussion using the LLM.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ...services import get_llm_service
from ..state import AgentState

logger = logging.getLogger(__name__)

SUMMARIZER_SYSTEM_PROMPT = """You are an expert meeting analyst. Your task is to create a concise, 
high-level summary of a meeting transcript.

Focus on:
1. The main purpose and objectives of the meeting
2. Key topics discussed
3. Important points raised by participants
4. Overall outcome or conclusion

Guidelines:
- Be concise but comprehensive
- Use bullet points for key topics
- Mention participant names when relevant
- Keep the summary to 3-5 paragraphs maximum
- Identify 3-7 key topics discussed

Output format:
## Summary
[Your concise summary paragraph]

## Key Topics
- Topic 1
- Topic 2
- Topic 3
..."""

SUMMARIZER_USER_PROMPT = """Please analyze and summarize the following meeting transcript:

Meeting Title: {title}
Participants: {participants}

--- TRANSCRIPT ---
{transcript}
--- END TRANSCRIPT ---

Provide a high-level summary and extract the key topics discussed."""


async def summarizer_node(state: AgentState) -> dict:
    """
    LangGraph node that generates meeting summaries.
    
    This node:
    1. Takes the parsed transcript
    2. Uses the LLM to generate a summary
    3. Extracts key topics from the meeting
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state with summary and key_topics
    """
    logger.info(f"Generating summary for meeting: {state['meeting_id']}")
    
    try:
        llm_service = get_llm_service()
        
        # Use parsed transcript if available, otherwise raw
        transcript = state.get("parsed_transcript") or state["raw_transcript"]
        participants = ", ".join(state.get("participants", [])) or "Not specified"
        
        # Create the prompt
        user_message = SUMMARIZER_USER_PROMPT.format(
            title=state["meeting_title"],
            participants=participants,
            transcript=transcript[:15000],  # Limit transcript length
        )
        
        messages = [
            SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        
        # Generate summary
        response = await llm_service.chat(messages)
        
        # Parse key topics from response
        key_topics = _extract_key_topics(response)
        
        logger.info(f"Summary generated, extracted {len(key_topics)} key topics")
        
        return {
            "summary": response,
            "key_topics": key_topics,
        }
    
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return {
            "error": f"Failed to generate summary: {str(e)}",
            "summary": None,
            "key_topics": [],
        }


def _extract_key_topics(summary_text: str) -> list[str]:
    """
    Extract key topics from the summary response.
    
    Args:
        summary_text: The LLM's summary response
    
    Returns:
        List of key topics
    """
    topics = []
    in_topics_section = False
    
    for line in summary_text.split("\n"):
        line = line.strip()
        
        # Check for topics section header
        if "key topics" in line.lower() or "topics discussed" in line.lower():
            in_topics_section = True
            continue
        
        # Check for end of topics section
        if in_topics_section and line.startswith("#"):
            break
        
        # Extract bullet points
        if in_topics_section and (line.startswith("-") or line.startswith("•") or line.startswith("*")):
            topic = line.lstrip("-•* ").strip()
            if topic:
                topics.append(topic)
    
    return topics
