"""API Dependencies Injection"""

from Core.Schemas import AgentConfig, LLMConfig
from .services import AgentManager
import os


def get_default_agent_config() -> AgentConfig:
    """Provides the default Agent config for the pipeline.
    
    In a real application, you would load these parameters correctly from 
    an environment variable or settings file rather than hardcoding.
    
    Returns:
        AgentConfig: The default configuration.
    """
    # Assuming standard defaults, using placeholder key as done in test.py
    llm_config = LLMConfig(
        api_base=os.getenv("LLM_API_BASE", "https://api.openai.com/v1/"),
        api_key=os.getenv("LLM_API_KEY", "your_api_key_here"),
    )
    return AgentConfig(llm=llm_config)


# Singleton manager instance
_agent_manager = AgentManager(default_config=get_default_agent_config())


def get_agent_manager() -> AgentManager:
    """Dependency injection wrapper for the AgentManager.
    
    Returns:
        AgentManager: The global singleton AgentManager instance.
    """
    return _agent_manager
