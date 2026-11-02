"""
True ADK Sub-agents for Waiver Extraction.

This module contains LlmAgent-based sub-agents that work with ADK's
orchestration, context passing, and streaming capabilities.

Architecture:
    orchestrator_agent (root)
        ├── overview_agent      - Initial field extraction from PDF
        ├── critic_agent        - AcroForm-based verification
        ├── reextraction_agent  - Re-extract failed fields
        └── normalisation_agent - Normalize output and build evidence

Analytics (processing stats, error analysis, evidence lookups) is a
separate, standalone ADK app - see analytics_agent/ at the project root,
not a sub-agent here.

Usage:
    from waiver_level_extraction_agent.adk_agents import root_agent
    # Use with: adk web waiver_level_extraction_agent
"""

from .orchestrator_adk import root_agent
from .overview_adk import overview_agent
from .critic_adk import critic_agent
from .reextraction_adk import reextraction_agent
from .normalisation_adk import normalisation_agent

__all__ = [
    "root_agent",
    "overview_agent",
    "critic_agent",
    "reextraction_agent",
    "normalisation_agent",
]
