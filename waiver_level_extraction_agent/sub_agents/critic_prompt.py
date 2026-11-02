"""
Critic agent prompt template.

Note: The main critique prompt is now in critic_agent.py (LLM-based).
This file provides formatting utilities for displaying critique results.
"""


def format_critique_summary(critique: dict) -> str:
    """
    Format a critique dict into a human-readable summary.

    Args:
        critique: The critique dict from CriticAgent.critique()

    Returns:
        Formatted string summary
    """
    lines = []
    lines.append(f"=== Critique Summary ===")
    lines.append(f"Accuracy: {critique.get('accuracy', 0):.1%}")
    lines.append(f"Decision: {critique.get('decision', 'unknown').upper()}")
    lines.append(f"")

    passing = critique.get("passing", [])
    failing = critique.get("failing", [])
    low_confidence = critique.get("low_confidence", [])

    lines.append(f"PASSING ({len(passing)} fields):")
    for f in passing[:5]:
        lines.append(f"  ✓ {f['field']}: {f.get('match_type', 'unknown')}")
    if len(passing) > 5:
        lines.append(f"  ... and {len(passing) - 5} more")

    lines.append(f"")
    lines.append(f"FAILING ({len(failing)} fields):")
    for f in failing:
        lines.append(f"  ✗ {f['field']}")
        lines.append(f"    Extracted: {f.get('extracted')}")
        lines.append(f"    Expected: {f.get('expected')}")
        lines.append(f"    Match type: {f.get('match_type')}")
        lines.append(f"    Reason: {f.get('reason')}")

    lines.append(f"")
    lines.append(f"LOW CONFIDENCE ({len(low_confidence)} fields):")
    for f in low_confidence:
        lines.append(f"  ? {f['field']}")
        lines.append(f"    Extracted: {f.get('extracted')}")
        lines.append(f"    Expected: {f.get('expected')}")
        lines.append(f"    Reason: {f.get('reason')}")

    return "\n".join(lines)


def format_three_bucket_breakdown(critique: dict) -> dict:
    """
    Format critique into a three-bucket breakdown for display.

    Args:
        critique: The critique dict from CriticAgent.critique()

    Returns:
        Dict with formatted breakdown
    """
    passing = critique.get("passing", [])
    failing = critique.get("failing", [])
    low_confidence = critique.get("low_confidence", [])

    return {
        "pass": {
            "count": len(passing),
            "fields": [
                {
                    "field": f["field"],
                    "match_type": f.get("match_type", "unknown"),
                }
                for f in passing
            ],
        },
        "fail": {
            "count": len(failing),
            "fields": [
                {
                    "field": f["field"],
                    "match_type": f.get("match_type", "unknown"),
                    "reason": f.get("reason", ""),
                    "extracted": f.get("extracted"),
                    "expected": f.get("expected"),
                }
                for f in failing
            ],
        },
        "low_confidence": {
            "count": len(low_confidence),
            "fields": [
                {
                    "field": f["field"],
                    "match_type": f.get("match_type", "unknown"),
                    "reason": f.get("reason", ""),
                    "extracted": f.get("extracted"),
                    "expected": f.get("expected"),
                }
                for f in low_confidence
            ],
        },
        "accuracy": critique.get("accuracy", 0),
        "decision": critique.get("decision", "unknown"),
    }


def format_unresolved_fields(
    field_name: str,
    attempt1_extracted,
    attempt1_reason: str,
    attempt2_extracted,
    attempt2_reason: str,
    expected_value
) -> dict:
    """
    Format an unresolved field with full history.

    Args:
        field_name: The field name
        attempt1_extracted: Value from Attempt 1
        attempt1_reason: Critic reason from Attempt 1
        attempt2_extracted: Value from Attempt 2 (re-extraction)
        attempt2_reason: Critic reason from Attempt 2
        expected_value: Ground truth value

    Returns:
        Dict with full unresolved field history
    """
    return {
        "field": field_name,
        "expected": expected_value,
        "attempts": [
            {
                "attempt": 1,
                "extracted": attempt1_extracted,
                "reason": attempt1_reason,
            },
            {
                "attempt": 2,
                "extracted": attempt2_extracted,
                "reason": attempt2_reason,
            },
        ],
    }
