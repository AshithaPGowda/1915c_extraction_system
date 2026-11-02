"""
ADK Sub-agents for Services Extraction.

Architecture:
    services_extraction_agent (root, aka orchestrator)
        |-- service_extraction_agent   - extracts one service at a time
        |-- services_critic_agent      - verifies + corrects + resolves
        |-- services_reextraction_agent - re-extracts flagged services
        |-- services_normalisation_agent - normalises + saves final output

Every tool behind these agents is a thin wrapper around the real functions
in waiver_level_extraction_agent/core/services_pipeline.py and
sub_agents/services_*.py - nothing is reimplemented.
"""

from .orchestrator_adk import root_agent
from .extraction_adk import extraction_agent
from .critic_adk import critic_agent
from .reextraction_adk import reextraction_agent
from .normalisation_adk import normalisation_agent

__all__ = [
    "root_agent",
    "extraction_agent",
    "critic_agent",
    "reextraction_agent",
    "normalisation_agent",
]
