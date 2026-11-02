"""
Waiver Overview Sub-Agent
Extracts general waiver information from a 1915(c) HCBS waiver PDF.
Uses layout_waiver_overview.json as the field schema.
Critically: outputs the list of service names found in Appendix C
so the Services agent can dynamically build its extraction prompt.
"""

import os
import json
from google.adk.agents import Agent
from ..shared_libraries.file_utils import log_event, write_json_output
from .overview_prompt import OVERVIEW_AGENT_PROMPT


def get_overview_layout() -> dict:
    """Load the waiver overview layout schema. Lazy load — called at runtime."""
    layout_path = os.path.join(
        os.path.dirname(__file__),
        "..", "data", "layouts", "layout_waiver_overview.json"
    )
    with open(layout_path) as f:
        return json.load(f)


overview_agent = Agent(
    name="waiver_overview_agent",
    model="gemini-2.5-flash",
    description=(
        "Extracts general waiver information from a 1915(c) HCBS waiver PDF. "
        "Returns structured JSON with all overview fields plus discovered service names."
    ),
    instruction=OVERVIEW_AGENT_PROMPT,
    tools=[],
)
