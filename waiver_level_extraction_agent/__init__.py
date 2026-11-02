"""
1915(c) HCBS Waiver Extraction Agent

ADK agent for extracting structured data from Medicaid waiver PDF documents.

Modes:
    - adk: True ADK sub-agents with orchestration (default)
    - pipeline: Python pipeline wrapped as tools

Set WAIVER_EXTRACTION_MODE environment variable to switch.
"""

from .agent import root_agent as root_agent
