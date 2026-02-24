# src/rule_builder/openai_client.py
from dotenv import load_dotenv
load_dotenv()
import os
import json
import time
from openai import OpenAI

from .prompt_template import RULE_PROMPT

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def call_llm(text, retry=3):

    prompt = RULE_PROMPT.format(text=text)

    for attempt in range(retry):

        try:

            response = client.chat.completions.create(

                model="gpt-4o-mini",

                messages=[
                    {"role": "user", "content": prompt}
                ],

                temperature=0
            )

            content = response.choices[0].message.content

            return json.loads(content)


        except Exception as e:

            print(f"LLM error: {e}")

            time.sleep(2)


    raise RuntimeError("LLM failed after retries")