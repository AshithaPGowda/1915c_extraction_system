"""
Output saving and logging tools for ADK agents.
"""

from ...shared_libraries.file_utils import write_json_output, log_event


def save_extraction_output(waiver_key: str, extracted_data: dict) -> dict:
    """Save extraction output to the outputs directory.

    Args:
        waiver_key: Waiver identifier
        extracted_data: The extracted fields

    Returns:
        dict with output path
    """
    try:
        # Separate data from metadata
        data = {k: v for k, v in extracted_data.items() if not k.startswith("_")}

        output = {
            "waiver_key": waiver_key,
            "extracted_fields": data,
        }

        path = write_json_output(waiver_key, "overview.json", output)

        return {
            "status": "success",
            "output_path": path,
            "field_count": len(data),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def log_agent_event(waiver_key: str, agent_name: str, action: str, details: dict) -> dict:
    """Log an event from an ADK agent.

    Args:
        waiver_key: Waiver identifier
        agent_name: Name of the agent logging
        action: Action being performed
        details: Additional details dict

    Returns:
        dict confirming log
    """
    try:
        log_event(waiver_key, {
            "agent": agent_name,
            "action": action,
            "mode": "adk",
            **details,
        })
        return {"status": "logged"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
