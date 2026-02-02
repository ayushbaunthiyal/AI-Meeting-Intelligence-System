"""
LLM Service for AI Meeting Intelligence System

This service provides a unified interface to OpenAI's language models
for chat completions and structured output generation.
"""

import logging
from typing import Optional, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ..config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMService:
    """
    Service for interacting with OpenAI language models.
    
    Provides methods for chat completions, structured output,
    and streaming responses.
    
    Attributes:
        llm: The ChatOpenAI instance
        model_name: Name of the model being used
    """
    
    _instance: Optional["LLMService"] = None
    _llm: Optional[ChatOpenAI] = None
    
    def __new__(cls) -> "LLMService":
        """Singleton pattern to reuse LLM instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the LLM service with OpenAI."""
        if self._llm is None:
            settings = get_settings()
            self.model_name = settings.openai_model
            
            logger.info(f"Initializing LLM with model: {self.model_name}")
            
            self._llm = ChatOpenAI(
                model=self.model_name,
                api_key=settings.openai_api_key,
                temperature=0.1,  # Low temperature for consistent outputs
            )
            
            logger.info("LLM service initialized successfully")
    
    @property
    def llm(self) -> ChatOpenAI:
        """Get the ChatOpenAI instance."""
        if self._llm is None:
            raise RuntimeError("LLM not initialized")
        return self._llm
    
    async def chat(
        self,
        messages: list[BaseMessage],
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send messages to the LLM and get a response.
        
        Args:
            messages: List of chat messages
            temperature: Optional temperature override
        
        Returns:
            The assistant's response text
        """
        llm = self.llm
        if temperature is not None:
            llm = llm.with_config({"temperature": temperature})
        
        response = await llm.ainvoke(messages)
        return response.content if isinstance(response.content, str) else str(response.content)
    
    async def chat_with_prompt(
        self,
        prompt_template: str,
        variables: dict,
        system_message: Optional[str] = None,
    ) -> str:
        """
        Chat using a prompt template with variables.
        
        Args:
            prompt_template: Template string with {variable} placeholders
            variables: Dictionary of variables to fill in the template
            system_message: Optional system message
        
        Returns:
            The assistant's response text
        """
        messages: list[BaseMessage] = []
        
        if system_message:
            messages.append(SystemMessage(content=system_message))
        
        # Format the prompt
        formatted_prompt = prompt_template.format(**variables)
        messages.append(HumanMessage(content=formatted_prompt))
        
        return await self.chat(messages)
    
    async def generate_structured(
        self,
        messages: list[BaseMessage],
        output_schema: type[T],
    ) -> T:
        """
        Generate a structured response using a Pydantic model.
        
        Args:
            messages: List of chat messages
            output_schema: Pydantic model class for the output
        
        Returns:
            Instance of the output schema
        """
        structured_llm = self.llm.with_structured_output(output_schema)
        response = await structured_llm.ainvoke(messages)
        return response
    
    def create_messages(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        chat_history: Optional[list[tuple[str, str]]] = None,
    ) -> list[BaseMessage]:
        """
        Create a list of messages for the LLM.
        
        Args:
            user_message: The user's current message
            system_message: Optional system message
            chat_history: Optional list of (role, content) tuples
        
        Returns:
            List of BaseMessage objects
        """
        messages: list[BaseMessage] = []
        
        if system_message:
            messages.append(SystemMessage(content=system_message))
        
        if chat_history:
            for role, content in chat_history:
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        
        messages.append(HumanMessage(content=user_message))
        
        return messages


def get_llm_service() -> LLMService:
    """
    Get the LLM service instance.
    
    Returns:
        LLMService: Singleton instance of the LLM service
    """
    return LLMService()
