"""一次性脚本：调一个极小 prompt，确认 .env 配置的模型在服务端真实可用。"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
load_dotenv()

from src.core.llm_logging import log_model_once  # noqa: E402

model = os.getenv("MINIMAX_MODEL", "<unset>")
base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
api_key = os.getenv("MINIMAX_API_KEY")

print(f"Requested model : {model}")
print(f"Base URL        : {base_url}")
print(f"API key present : {bool(api_key)}")

if not api_key:
    sys.exit("MINIMAX_API_KEY missing in .env")

client = OpenAI(api_key=api_key, base_url=base_url)
try:
    resp = client.chat.completions.create(
        model=model,
        max_tokens=16,
        messages=[{"role": "user", "content": "Say only the word: pong"}],
    )
except Exception as e:
    print(f"\n[ERROR] API call failed: {type(e).__name__}: {e}")
    sys.exit(1)

log_model_once("check_model.py", model, resp)
print(f"\nServer returned model : {getattr(resp, 'model', '<none>')}")
print(f"Reply                 : {resp.choices[0].message.content!r}")
if resp.usage:
    print(f"Tokens                : in={resp.usage.prompt_tokens} out={resp.usage.completion_tokens}")
