"""
Rule DSL Validator and Parser
"""
import json
from jsonschema import validate, ValidationError, Draft202012Validator
from pathlib import Path
from typing import Dict, Any, Optional

from src.models.rule import Rule, RuleConditions, RuleActions, ROI


SCHEMA_PATH = Path("configs/schemas/rule_dsl_schema.json")


def load_schema() -> Dict:
    """Load JSON Schema"""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")
    with SCHEMA_PATH.open('r', encoding='utf-8') as f:
        return json.load(f)


def validate_rule_json(rule_data: Dict) -> tuple[bool, Optional[str]]:
    """
    Validate rule JSON against schema + additional business rules.
    Returns (is_valid, error_message)
    """
    try:
        schema = load_schema()
        validator = Draft202012Validator(schema)
        validator.validate(rule_data)

        # Additional custom validation (operator allowlist, value range, etc.)
        method = rule_data.get("method")
        if method not in ["direct", "composite"]:
            return False, f"Invalid method: {method}. Must be 'direct' or 'composite'"

        conditions = rule_data.get("conditions", {})
        if "dwell_seconds" in conditions and not (0.1 <= conditions["dwell_seconds"] <= 300):
            return False, "dwell_seconds must be between 0.1 and 300"

        # Có thể thêm nhiều rule khác: chỉ cho phép require_helmetless nếu method=composite, v.v.

        return True, None

    except ValidationError as e:
        return False, f"JSON Schema validation failed: {e.message}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def parse_rule_from_json(rule_data: Dict) -> Rule:
    """
    Parse validated JSON data into Rule object (tương thích với code hiện tại)
    """
    if not validate_rule_json(rule_data)[0]:
        raise ValueError("Invalid rule JSON")

    # Parse ROI
    roi = None
    roi_data = rule_data["conditions"].get("inside_roi")
    if roi_data and roi_data.get("enabled", False):
        roi = ROI(
            enabled=True,
            roi_type=roi_data["type"],
            points=roi_data["points"]
        )

    # Parse conditions (bao gồm require_helmetless)
    conditions_data = rule_data["conditions"]
    conditions = RuleConditions(
        dwell_seconds=conditions_data["dwell_seconds"],
        min_confidence=conditions_data["min_confidence"],
        min_frames=conditions_data.get("min_frames", 3),
        require_helmetless=conditions_data.get("require_helmetless", False)
    )

    # Parse actions
    actions_data = rule_data["actions"]
    actions = RuleActions(
        cooldown_seconds=actions_data["cooldown_seconds"],
        record_pre_seconds=actions_data.get("record_pre_seconds", 0),
        record_post_seconds=actions_data.get("record_post_seconds", 0),
        notify_channels=actions_data["notify_channels"]
    )

    return Rule(
        rule_id=rule_data["rule_id"],
        area_id=rule_data.get("area_id", "default"),
        description=rule_data["description"],
        prompt_positive=rule_data["detection"]["prompt_positive"],
        prompt_negative=rule_data["detection"].get("prompt_negative"),
        box_threshold=rule_data["detection"]["box_threshold"],
        text_threshold=rule_data["detection"]["text_threshold"],
        conditions=conditions,
        roi=roi,
        actions=actions,
        metadata=rule_data.get("metadata", {})
    )