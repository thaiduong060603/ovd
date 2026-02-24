import json
from pathlib import Path
from datetime import datetime

from .openai_client import call_llm
from .cache import load_cache, save_cache

from src.utils.rule_validator import validate_rule_json


RULE_DIR = Path("configs/rules")
RULE_DIR.mkdir(parents=True, exist_ok=True)


def generate_rule_file(text):

    print("Phase 5: Generating rule from text...")

    cache = load_cache(text)

    if cache:

        print("Using cached rule")

        rule_json = cache

    else:

        rule_json = call_llm(text)

        is_valid, error = validate_rule_json(rule_json)

        if not is_valid:

            raise ValueError(error)

        save_cache(text, rule_json)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = RULE_DIR / f"generated_{timestamp}.json"


    with open(path, "w") as f:

        json.dump(rule_json, f, indent=2)


    print(f"Rule saved: {path}")

    return str(path)