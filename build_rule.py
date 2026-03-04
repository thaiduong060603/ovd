"""
Usage:
    python build_rule.py "Person without vest in warehouse" --name no_vest_warehouse
    python build_rule.py "..." --auto-approve
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rule_builder.rule_builder import generate_rule_file


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Generate OVD monitoring rule from natural language description"
    )
    parser.add_argument(
        "description",
        help='Natural language rule description, e.g. "Alert when person without helmet enters zone A"'
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output filename (without .json). Default: <rule_id>_<timestamp>.json"
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip human approval step (for testing/CI)"
    )
    args = parser.parse_args()

    try:
        path = generate_rule_file(
            text=args.description,
            auto_approve=args.auto_approve,
            output_name=args.name,
        )
        print(f"\nDone. Use with main.py:")
        print(f"  python main.py --input <video> --rule {path}")
    except ValueError as e:
        print(f"\n✗ {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n✗ LLM Error: {e}")
        print("  Check your GEMINI_API_KEY in .env file")
        sys.exit(1)


if __name__ == "__main__":
    main()