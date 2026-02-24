RULE_PROMPT = """
You are an OVD rule generator.

Convert the following description into JSON DSL rule.

Description:
"{text}"

Return ONLY valid JSON.

Schema:

{{
  "rule_id": "string",
  "name": "string",
  "description": "string",
  "event": {{
    "type": "presence",
    "object": "string",
    "zone": "string",
    "duration": number
  }},
  "actions": {{
    "record": true,
    "notify": true
  }}
}}
"""