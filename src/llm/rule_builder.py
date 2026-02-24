"""
Phase 5: LLM Rule Builder using Google Gemini - Convert natural language to JSON DSL with human approval
"""
import json
from typing import Dict, List, Optional
from pathlib import Path
import argparse
#import google.generativeai as genai  # Nếu dùng package cũ, giữ nguyên
import google.genai as genai

from src.utils.rule_validator import validate_rule_json


# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyDfwIVlyaQLyrJG1Z0PuBYYqGO9an2fZy0"  # Thay bằng key thật
genai.configure(api_key=GEMINI_API_KEY)

SCHEMA_PATH = Path("configs/schemas/rule_dsl_schema.json")
APPROVED_DIR = Path("configs/rules/approved")
AVAILABLE_ZONES = ["entrance_zone_1", "factory_back", "forklift_area"]
ALLOWED_METHODS = ["direct", "composite"]
ALLOWED_CHANNELS = ["console", "email", "slack", "mqtt"]


def load_schema() -> Dict:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")
    with SCHEMA_PATH.open('r', encoding='utf-8') as f:
        return json.load(f)


def generate_rule_proposal(natural_text: str, temperature: float = 0.3, max_retries: int = 3) -> Dict:
    """Gọi Gemini để tạo JSON DSL đề xuất, retry nếu parse lỗi"""
    schema_str = json.dumps(load_schema(), indent=2)

    system_prompt = f"""
Bạn là trợ lý chuyên xây dựng rule giám sát an toàn cho hệ thống OVD.

Nhiệm vụ:
- Chuyển mô tả tự nhiên thành JSON DSL **chính xác theo schema** sau.
- Chỉ trả về JSON thuần túy, KHÔNG có text giải thích, KHÔNG có code block, KHÔNG có dấu ```json.

Schema:
{schema_str}

Quy tắc nghiêm ngặt:
- Nếu mô tả rõ ràng → trả về 1 object JSON duy nhất.
- Nếu mơ hồ (zone, method, threshold, v.v.) → trả về object với:
  "candidates": mảng các JSON đề xuất khác nhau
  "uncertainty_notes": chuỗi ghi chú phần không rõ
- Zone phải từ danh sách: {', '.join(AVAILABLE_ZONES)}
- Method chỉ: "direct" hoặc "composite"
- Notify channels chỉ: {', '.join(ALLOWED_CHANNELS)}
- Dwell_seconds thường 1.0–5.0, cooldown 30–300
- Rule_id snake_case unique (ví dụ helmetless_entrance_2s)

Mô tả người dùng:
{natural_text}
"""

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json"  # Gemini ưu tiên trả JSON
        }
    )

    for attempt in range(max_retries):
        try:
            response = model.generate_content(system_prompt)
            content = response.text.strip()

            # Clean content (Gemini đôi khi thêm thừa)
            content = content.replace("```json", "").replace("```", "").strip()

            parsed = json.loads(content)
            return parsed

        except json.JSONDecodeError as e:
            print(f"Attempt {attempt+1}/{max_retries}: JSON parse error - {str(e)}")
            print("Raw content:", content)
            continue
        except Exception as e:
            return {"error": f"Gemini error: {str(e)}"}

    return {"error": f"Failed after {max_retries} attempts to get valid JSON"}


def human_review_and_approve(proposed: Dict) -> Optional[Dict]:
    """CLI review và approve rule"""
    print("\n" + "="*80)
    print("PROPOSED RULE FROM GEMINI")
    print("="*80)

    if "error" in proposed:
        print("LLM ERROR:", proposed["error"])
        return None

    if "candidates" in proposed:
        print("Multiple candidates proposed due to uncertainty:")
        for i, cand in enumerate(proposed["candidates"], 1):
            print(f"\nCandidate {i}:")
            print(json.dumps(cand, indent=2, ensure_ascii=False))
        print("\nUncertainty notes:", proposed.get("uncertainty_notes", "None"))
        selected = input("\nChọn candidate số (1/2/...) hoặc 'cancel': ").strip()
        if selected.isdigit() and 1 <= int(selected) <= len(proposed["candidates"]):
            selected_rule = proposed["candidates"][int(selected)-1]
        else:
            print("Cancel.")
            return None
    else:
        print(json.dumps(proposed, indent=2, ensure_ascii=False))
        selected_rule = proposed

    choice = input("\nApprove rule này? (y/n/edit/cancel): ").strip().lower()

    if choice == 'y':
        # Validate lại trước khi lưu
        is_valid, msg = validate_rule_json(selected_rule)
        if not is_valid:
            print("Validation failed:", msg)
            return None
        return selected_rule

    elif choice == 'edit':
        print("\nDán JSON đã chỉnh sửa (Ctrl+C để hủy):")
        try:
            edited_text = input()
            edited_rule = json.loads(edited_text)
            is_valid, msg = validate_rule_json(edited_rule)
            if not is_valid:
                print("Validation failed:", msg)
                return None
            return edited_rule
        except Exception as e:
            print("Edit invalid:", str(e))
            return None

    else:
        print("Rule bị từ chối.")
        return None


def save_approved_rule(rule_data: Dict):
    APPROVED_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{rule_data['rule_id']}.json"
    path = APPROVED_DIR / filename

    with path.open('w', encoding='utf-8') as f:
        json.dump(rule_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Rule đã được approve và lưu tại: {path}")
    print("Để sử dụng rule này, chạy:")
    print(f"python main.py --rule {path} --input <video> --display")


def main():
    parser = argparse.ArgumentParser(description="Phase 5: LLM Rule Builder with Gemini")
    parser.add_argument("description", nargs='+', help="Mô tả rule bằng ngôn ngữ tự nhiên")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature cho LLM (thấp hơn = chính xác hơn)")
    args = parser.parse_args()

    natural_text = " ".join(args.description)
    print(f"\nXây dựng rule từ mô tả: {natural_text}")

    proposed = generate_rule_proposal(natural_text, temperature=args.temperature)

    approved_rule = human_review_and_approve(proposed)

    if approved_rule:
        save_approved_rule(approved_rule)
    else:
        print("\nKhông có rule nào được approve.")


if __name__ == "__main__":
    main()