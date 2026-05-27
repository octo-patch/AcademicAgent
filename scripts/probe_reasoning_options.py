"""Test reasoning-model handling options on MiniMax-M2:
  (A) larger max_tokens — does it complete <think> + valid JSON?
  (B) try common 'disable thinking' parameters
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"),
)

MODEL = "MiniMax-M2"
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ping_reply",
        "schema": {
            "type": "object",
            "properties": {"word": {"type": "string"}},
            "required": ["word"],
            "additionalProperties": False,
        },
    },
}


def call(label: str, **extra) -> None:
    print(f"\n=== {label} ===")
    try:
        r = client.chat.completions.create(
            model=MODEL,
            max_tokens=extra.pop("max_tokens", 4000),
            messages=[{"role": "user", "content": "Return JSON with a single field 'word' = 'pong'"}],
            response_format=SCHEMA,
            **extra,
        )
        content = r.choices[0].message.content or ""
        print(f"  in/out tokens : {r.usage.prompt_tokens}/{r.usage.completion_tokens}")
        print(f"  has <think>   : {'<think>' in content}")
        print(f"  has </think>  : {'</think>' in content}")
        # Try to extract JSON after </think>
        if "</think>" in content:
            after = content.split("</think>", 1)[1].strip()
            print(f"  after </think>: {after[:200]!r}")
            try:
                parsed = json.loads(after)
                print(f"  JSON parse    : OK -> {parsed}")
            except Exception as e:
                print(f"  JSON parse    : FAIL: {e}")
        else:
            print(f"  raw (first 200): {content[:200]!r}")
            try:
                parsed = json.loads(content)
                print(f"  JSON parse    : OK -> {parsed}")
            except Exception as e:
                print(f"  JSON parse    : FAIL: {e}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")


# (A) Generous max_tokens
call("A1: max_tokens=2000 (default-ish)", max_tokens=2000)
call("A2: max_tokens=8000 (generous)",   max_tokens=8000)

# (B) Try disable-thinking flags (one at a time; ignore errors if param unsupported)
call("B1: extra_body enable_thinking=False", extra_body={"enable_thinking": False})
call("B2: extra_body thinking_budget=0",      extra_body={"thinking_budget": 0})
call("B3: extra_body reasoning_effort=none",  extra_body={"reasoning_effort": "none"})
call("B4: extra_body chat_template_kwargs enable_thinking=False",
     extra_body={"chat_template_kwargs": {"enable_thinking": False}})
