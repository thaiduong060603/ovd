"""
prompt_generator.py — LLM-based multilingual prompt and rule generator.

Converts user instructions (Japanese / English / Vietnamese / any language)
into structured NanoOWL prompts + RuleEngineV1-compatible rule parameters.

Prompt rule enforced: ALL prompts follow "a photo of <object>" pattern.

LLM backends (auto-selected in priority order):
  1. Ollama  — local, recommended on Jetson (ollama pull gemma3:4b)
  2. Gemini  — cloud fallback (set GEMINI_API_KEY env var)
  3. Keyword fallback — no LLM required, covers common cases

Install Ollama on Jetson:
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull gemma3:4b
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Output data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AttributeCheck:
    class_name: str    # detected class to check, e.g. "a photo of helmet"
    attribute:  str    # "color" (extendable to "size", "shape", …)
    value:      str    # expected value, e.g. "yellow"

    def to_dict(self) -> dict:
        return {"class_name": self.class_name, "attribute": self.attribute, "value": self.value}


@dataclass
class GeneratedRule:
    """Structured output from the LLM prompt generator."""

    prompts:            List[str]
    rule_type:          str                 # roi_entry | dwell_time | helmet_safety | proximity_danger
    description:        str
    dwell_seconds:      float               = 0.0
    min_confidence:     float               = 0.35
    min_frames:         int                 = 3
    cooldown_seconds:   float               = 5.0
    roi_enabled:        bool                = True
    near_class:         Optional[str]       = None
    near_distance_px:   Optional[float]     = None
    require_helmetless: bool                = False
    attribute_checks:   List[AttributeCheck] = field(default_factory=list)
    raw_llm_output:     str                 = ""

    # ── YAML export ───────────────────────────────────────────────────────────

    def to_rule_yaml(self) -> str:
        """Return YAML string compatible with Rule.from_yaml()."""
        lines = [
            f"rule_id: rule_{self.rule_type}",
            f"description: {self.description}",
            "detection:",
            f"  prompt_positive: {', '.join(self.prompts)}",
            "  box_threshold: 0.15",
            "  text_threshold: 0.15",
            "conditions:",
            f"  dwell_seconds: {self.dwell_seconds}",
            f"  min_confidence: {self.min_confidence}",
            f"  min_frames: {self.min_frames}",
        ]
        if self.require_helmetless:
            lines.append("  require_helmetless: true")
        if self.near_class:
            lines.append(f"  near_class: {self.near_class}")
        if self.near_distance_px is not None:
            lines.append(f"  near_distance_px: {self.near_distance_px}")
        lines += [
            "actions:",
            f"  cooldown_seconds: {self.cooldown_seconds}",
            "  record_pre_seconds: 5.0",
            "  record_post_seconds: 5.0",
            "  notify_channels: []",
            "roi:",
            f"  enabled: {'true' if self.roi_enabled else 'false'}",
            "  type: polygon",
            "  points: []",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "prompts":            self.prompts,
            "rule_type":          self.rule_type,
            "description":        self.description,
            "dwell_seconds":      self.dwell_seconds,
            "min_confidence":     self.min_confidence,
            "min_frames":         self.min_frames,
            "cooldown_seconds":   self.cooldown_seconds,
            "roi_enabled":        self.roi_enabled,
            "near_class":         self.near_class,
            "near_distance_px":   self.near_distance_px,
            "require_helmetless": self.require_helmetless,
            "attribute_checks":   [a.to_dict() for a in self.attribute_checks],
            "rule_yaml":          self.to_rule_yaml(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# System prompt for LLM
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a computer vision rule generation assistant for an industrial safety monitoring system.
Your task: convert a user's monitoring instruction (in ANY language) into structured JSON.

DETECTION PROMPT RULES — strictly follow these:
- Every prompt MUST use the pattern: "a photo of <object>"
- Keep objects SIMPLE and CONCRETE: 1-3 words after "a photo of"
- NEVER include color attributes inside prompts — use attribute_checks instead
  - WRONG: "a photo of yellow helmet"
  - RIGHT:  prompt="a photo of helmet", attribute_checks=[{"class_name":"a photo of helmet","attribute":"color","value":"yellow"}]
- Good: "a photo of person", "a photo of helmet", "a photo of forklift"
- Bad:  "a photo of person wearing equipment", "a photo of dangerous area"

RULE TYPES:
- "roi_entry"         — alert immediately when object enters a zone
- "dwell_time"        — alert when object stays in zone > dwell_seconds
- "helmet_safety"     — alert when worker lacks helmet (set require_helmetless: true)
- "proximity_danger"  — alert when two object types come closer than near_distance_px

OUTPUT: respond with ONLY valid JSON (no markdown, no explanation):
{
  "prompts": ["a photo of <obj1>"],
  "rule_type": "<roi_entry|dwell_time|helmet_safety|proximity_danger>",
  "description": "<short English description>",
  "dwell_seconds": 0.0,
  "min_confidence": 0.35,
  "min_frames": 3,
  "cooldown_seconds": 5.0,
  "roi_enabled": true,
  "near_class": null,
  "near_distance_px": null,
  "require_helmetless": false,
  "attribute_checks": []
}"""

_USER_TEMPLATE = "Generate a detection rule for: {user_input}"


# ─────────────────────────────────────────────────────────────────────────────
# PromptGenerator
# ─────────────────────────────────────────────────────────────────────────────

class PromptGenerator:
    """
    Converts multilingual monitoring instructions into structured NanoOWL rules.

    Example:
        gen = PromptGenerator()
        rule = gen.generate("黄色いヘルメットをかぶっていない作業者を検出")
        print(rule.to_rule_yaml())
    """

    def __init__(
        self,
        backend:        str            = "auto",
        model_name:     str            = "gemma3:4b",
        ollama_url:     str            = "http://localhost:11434",
        gemini_api_key: Optional[str]  = None,
    ):
        self.backend        = backend
        self.model_name     = model_name
        self.ollama_url     = ollama_url
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        self._active        = self._resolve_backend()

    # ── Backend resolution ────────────────────────────────────────────────────

    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if self._ollama_available():
            print("[LLM] Backend: Ollama")
            return "ollama"
        if self.gemini_api_key:
            print("[LLM] Backend: Gemini API")
            return "gemini"
        print("[LLM] Backend: keyword fallback (no LLM found)")
        return "fallback"

    def _ollama_available(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, user_input: str) -> GeneratedRule:
        """
        Convert a monitoring instruction in any language to a GeneratedRule.

        Args:
            user_input: free-text instruction (Japanese / English / Vietnamese / …)

        Returns:
            GeneratedRule with prompts, rule params, and attribute checks.
        """
        print(f"[LLM] Generating rule for: {user_input!r}")
        raw = ""
        try:
            if self._active == "ollama":
                raw = self._call_ollama(user_input)
            elif self._active == "gemini":
                raw = self._call_gemini(user_input)
            else:
                return self._fallback(user_input)

            rule = self._parse_response(raw)
            rule.raw_llm_output = raw
            print(f"[LLM] Done: prompts={rule.prompts}  rule_type={rule.rule_type}")
            return rule

        except Exception as exc:
            print(f"[LLM] Failed ({exc}), falling back to keyword matching.")
            fb = self._fallback(user_input)
            fb.raw_llm_output = raw
            return fb

    # ── LLM backends ─────────────────────────────────────────────────────────

    def _call_ollama(self, user_input: str) -> str:
        import requests
        payload = {
            "model":  self.model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _USER_TEMPLATE.format(user_input=user_input)},
            ],
            "options": {"temperature": 0.1, "top_p": 0.9},
        }
        r = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=90,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _call_gemini(self, user_input: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel("gemma-3-4b-it")
        full_prompt = _SYSTEM_PROMPT + "\n\n" + _USER_TEMPLATE.format(user_input=user_input)
        response = model.generate_content(full_prompt)
        return response.text

    # ── JSON parsing ──────────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> GeneratedRule:
        """Extract the first JSON object from LLM output."""
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in LLM output.")

        data: Dict[str, Any] = json.loads(m.group())

        attr_checks = [
            AttributeCheck(
                class_name = ac.get("class_name", ""),
                attribute  = ac.get("attribute", "color"),
                value      = ac.get("value", ""),
            )
            for ac in data.get("attribute_checks", [])
        ]

        raw_prompts: List[str] = data.get("prompts", ["a photo of person"])
        prompts = [self._normalise_prompt(p) for p in raw_prompts]

        return GeneratedRule(
            prompts            = prompts,
            rule_type          = data.get("rule_type", "roi_entry"),
            description        = data.get("description", "Custom monitoring rule"),
            dwell_seconds      = float(data.get("dwell_seconds", 0.0)),
            min_confidence     = float(data.get("min_confidence", 0.35)),
            min_frames         = int(data.get("min_frames", 3)),
            cooldown_seconds   = float(data.get("cooldown_seconds", 5.0)),
            roi_enabled        = bool(data.get("roi_enabled", True)),
            near_class         = data.get("near_class") or None,
            near_distance_px   = float(data["near_distance_px"]) if data.get("near_distance_px") else None,
            require_helmetless = bool(data.get("require_helmetless", False)),
            attribute_checks   = attr_checks,
        )

    @staticmethod
    def _normalise_prompt(prompt: str) -> str:
        """Enforce 'a photo of <X>' pattern."""
        p = prompt.strip()
        lower = p.lower()
        if lower.startswith("a photo of "):
            return p
        for prefix in ("photo of ", "image of ", "picture of ", "a ", "an "):
            if lower.startswith(prefix):
                p = p[len(prefix):]
                break
        return f"a photo of {p}"

    # ── Keyword fallback ──────────────────────────────────────────────────────

    def _fallback(self, user_input: str) -> GeneratedRule:
        """Keyword-based rule builder — works without any LLM."""
        text      = user_input.lower()
        prompts:     List[str]           = []
        attr_checks: List[AttributeCheck] = []
        rule_type = "roi_entry"
        desc      = f"Auto-generated rule for: {user_input[:60]}"

        # ── Detect color keywords ─────────────────────────────────────────────
        color_map = {
            "yellow": "yellow", "vàng": "yellow", "黄": "yellow", "黄色": "yellow",
            "red":    "red",    "đỏ":   "red",    "赤": "red",
            "blue":   "blue",   "xanh": "blue",   "青": "blue",
            "white":  "white",  "trắng":"white",  "白": "white",
            "orange": "orange", "cam":  "orange",
            "green":  "green",  "lục":  "green",  "緑": "green",
        }
        found_color: Optional[str] = None
        for kw, color in color_map.items():
            if kw in text:
                found_color = color
                break

        # ── Detect object keywords ────────────────────────────────────────────
        object_map = {
            "person":   ["person", "người", "人", "worker", "công nhân", "作業者", "人間"],
            "helmet":   ["helmet", "mũ", "ヘルメット", "hard hat", "hardhat", "安全帽", "mũ bảo hộ"],
            "forklift": ["forklift", "xe nâng", "フォークリフト"],
            "fire":     ["fire", "cháy", "火", "flame", "煙", "smoke"],
            "car":      ["car", "vehicle", "xe ô tô", "車", "ô tô"],
        }
        for obj, keywords in object_map.items():
            if any(kw in text for kw in keywords):
                prompt = f"a photo of {obj}"
                prompts.append(prompt)
                if found_color and obj == "helmet":
                    attr_checks.append(AttributeCheck(
                        class_name = prompt,
                        attribute  = "color",
                        value      = found_color,
                    ))

        # ── Detect rule type ──────────────────────────────────────────────────
        if any(k in text for k in ["dwell", "stay", "ở lại", "滞在", "居続ける", "lâu"]):
            rule_type = "dwell_time"
        elif any(k in text for k in ["helmet", "mũ", "ヘルメット", "hard hat", "mũ bảo hộ"]):
            rule_type = "helmet_safety"
        elif any(k in text for k in ["near", "close", "gần", "近く", "proximity", "forklift", "xe nâng"]):
            rule_type = "proximity_danger"

        if not prompts:
            prompts = ["a photo of person"]

        return GeneratedRule(
            prompts          = prompts,
            rule_type        = rule_type,
            description      = desc,
            dwell_seconds    = 5.0 if rule_type == "dwell_time" else 0.0,
            attribute_checks = attr_checks,
        )
