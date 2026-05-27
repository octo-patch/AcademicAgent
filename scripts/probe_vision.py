"""Empirical probe: which MiniMax models actually accept images via the
OpenAI-compatible HTTP endpoint (no MCP involved).

Method
------
1. Synthesize a tiny PNG with known content: the text "VLM-OK 42" plus a
   red filled circle on a white background. Both signals are easy to verify
   in the model reply.
2. Base64-encode the PNG and send it via the standard OpenAI multimodal
   message format (image_url content block with a data URI) — exactly what
   src/core/multimodal.py does today.
3. For each candidate model, record:
     - HTTP-level success (did the API accept the request at all?)
     - Whether the reply mentions "42" (text recognition)
     - Whether the reply mentions "red" / "circle" (visual recognition)
   "Real" vision support = both probes hit. Hallucination = HTTP OK but
   neither hit, or model refuses the image silently.

Run:
    python scripts/probe_vision.py
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

load_dotenv()


CANDIDATES = [
    "MiniMax-Text-01",
    "MiniMax-VL-01",
    "abab6.5s-chat",
    "abab6.5-chat",
    "MiniMax-M2",
    "MiniMax-M2.7",
]

PROMPT = (
    "Look at this image carefully. Reply in one short sentence: "
    "(a) what number is written, and (b) what colored shape is present."
)


def build_test_image() -> bytes:
    """Generate a 256x128 PNG with 'VLM-OK 42' text + a red filled circle."""
    img = Image.new("RGB", (256, 128), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    draw.text((12, 40), "VLM-OK 42", fill="black", font=font)

    draw.ellipse((180, 30, 240, 90), fill="red", outline="black")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def probe_model(client: OpenAI, model: str, image_b64: str) -> dict:
    out = {"model": model}
    t0 = time.time()
    data_uri = f"data:image/png;base64,{image_b64}"
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
        )
    except Exception as e:
        out["http_ok"] = False
        out["error"] = f"{type(e).__name__}: {str(e)[:240]}"
        out["latency_s"] = round(time.time() - t0, 2)
        return out

    reply = (resp.choices[0].message.content or "").strip()
    low = reply.lower()
    out["http_ok"] = True
    out["latency_s"] = round(time.time() - t0, 2)
    out["returned_model"] = getattr(resp, "model", None)
    out["reply"] = reply[:280]
    out["saw_number"] = "42" in reply
    out["saw_color"] = ("red" in low) and ("circle" in low or "ellipse" in low or "dot" in low or "shape" in low)
    if resp.usage:
        out["tokens"] = f"{resp.usage.prompt_tokens}/{resp.usage.completion_tokens}"
    return out


def verdict(r: dict) -> str:
    if not r.get("http_ok"):
        return "HTTP-FAIL"
    if r.get("saw_number") and r.get("saw_color"):
        return "VISION-OK"
    if r.get("saw_number") or r.get("saw_color"):
        return "PARTIAL"
    return "BLIND"  # endpoint accepted the call but did not read the image


def main() -> int:
    api_key = os.getenv("MINIMAX_API_KEY")
    base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
    if not api_key:
        print("MINIMAX_API_KEY missing in .env")
        return 1

    print(f"Endpoint: {base_url}")
    print(f"Transport: plain HTTP via OpenAI-compatible SDK (no MCP)\n")

    png_bytes = build_test_image()
    image_b64 = base64.b64encode(png_bytes).decode("ascii")
    # Save a copy so we can manually inspect what we sent
    artifact = Path(__file__).resolve().parent.parent / "data" / "_probe_vision.png"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(png_bytes)
    print(f"Test image written to {artifact} ({len(png_bytes)} bytes, b64 len={len(image_b64)})\n")

    client = OpenAI(api_key=api_key, base_url=base_url)

    header = f"{'Model':<24} {'Verdict':<10} {'#42':<4} {'red+shape':<10} {'Lat(s)':<7} {'Returned':<24}"
    print(header)
    print("-" * len(header))

    details = []
    for m in CANDIDATES:
        r = probe_model(client, m, image_b64)
        details.append(r)
        v = verdict(r)
        if r.get("http_ok"):
            print(
                f"{m:<24} {v:<10} "
                f"{('Y' if r.get('saw_number') else 'n'):<4} "
                f"{('Y' if r.get('saw_color') else 'n'):<10} "
                f"{r.get('latency_s','?'):<7} "
                f"{(r.get('returned_model') or '?')[:22]:<24}"
            )
        else:
            print(f"{m:<24} {v:<10} -    -          {r.get('latency_s','?'):<7} {r.get('error','')[:60]}")

    print("\nReplies:")
    for r in details:
        print(f"\n--- {r['model']} ({verdict(r)}) ---")
        if r.get("http_ok"):
            print(f"  reply: {r.get('reply','')!r}")
        else:
            print(f"  error: {r.get('error','')}")

    print(
        "\nLegend:\n"
        "  VISION-OK = both '42' and a red shape recognized => model genuinely reads images\n"
        "  PARTIAL   = one of two signals (often hallucinated; treat as unreliable)\n"
        "  BLIND     = HTTP accepted but neither signal => endpoint silently dropped/ignored image\n"
        "  HTTP-FAIL = endpoint rejected the request (model not in catalog, or no image support)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
