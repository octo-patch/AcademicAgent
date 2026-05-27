"""Capability probe: ping multiple MiniMax models, detect reasoning behavior + structured-output support.

For each model, runs 2 tiny calls:
  (1) plain  : "Say only the word: pong" with max_tokens=64 -> detect <think>, latency, returned model name
  (2) schema : same prompt but with response_format=json_schema -> verify structured-output works
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CANDIDATES = [
    "MiniMax-Text-01",
    "MiniMax-M2",
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
]

PROMPT = "Say only the word: pong"

JSON_SCHEMA = {
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


def probe(client: OpenAI, model: str) -> dict:
    result = {"model": model}
    # (1) plain
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": PROMPT}],
        )
        content = r.choices[0].message.content or ""
        result["plain_ok"] = True
        result["plain_returned_model"] = getattr(r, "model", None)
        result["plain_has_think"] = "<think>" in content
        result["plain_reply"] = content[:120]
        result["plain_in_tokens"] = r.usage.prompt_tokens if r.usage else None
        result["plain_out_tokens"] = r.usage.completion_tokens if r.usage else None
        result["plain_latency_s"] = round(time.time() - t0, 2)
    except Exception as e:
        result["plain_ok"] = False
        result["plain_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return result  # skip schema test if plain failed

    # (2) schema
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model,
            max_tokens=128,
            messages=[
                {"role": "user", "content": "Return JSON with a single field 'word' = 'pong'"},
            ],
            response_format=JSON_SCHEMA,
        )
        content = r.choices[0].message.content or ""
        result["schema_ok"] = True
        result["schema_reply"] = content[:200]
        result["schema_latency_s"] = round(time.time() - t0, 2)
    except Exception as e:
        result["schema_ok"] = False
        result["schema_error"] = f"{type(e).__name__}: {str(e)[:200]}"

    return result


def main() -> int:
    api_key = os.getenv("MINIMAX_API_KEY")
    base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
    if not api_key:
        print("MINIMAX_API_KEY missing")
        return 1
    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"Probing {len(CANDIDATES)} models against {base_url}\n")
    print(f"{'Model':<28} {'Plain':<6} {'Schema':<7} {'Reasoning':<10} {'Returned':<28} {'In/Out':<10} {'Lat(s)':<8}")
    print("-" * 110)
    details = []
    for m in CANDIDATES:
        r = probe(client, m)
        details.append(r)
        if not r.get("plain_ok"):
            print(f"{m:<28} FAIL   -       -          -                            -          -         {r.get('plain_error','')}")
            continue
        plain = "OK" if r.get("plain_ok") else "FAIL"
        schema = "OK" if r.get("schema_ok") else "FAIL"
        reasoning = "YES" if r.get("plain_has_think") else "no"
        returned = (r.get("plain_returned_model") or "?")[:26]
        toks = f"{r.get('plain_in_tokens','?')}/{r.get('plain_out_tokens','?')}"
        lat = f"{r.get('plain_latency_s','?')}"
        print(f"{m:<28} {plain:<6} {schema:<7} {reasoning:<10} {returned:<28} {toks:<10} {lat:<8}")

    print("\nDetailed replies:")
    for r in details:
        print(f"\n--- {r['model']} ---")
        if r.get("plain_ok"):
            print(f"  plain reply : {r.get('plain_reply','')!r}")
        else:
            print(f"  plain error : {r.get('plain_error','')}")
        if r.get("schema_ok"):
            print(f"  schema reply: {r.get('schema_reply','')!r}")
        elif "schema_error" in r:
            print(f"  schema error: {r.get('schema_error','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
