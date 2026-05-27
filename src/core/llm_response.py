"""LLM 响应清理工具。

MiniMax M2 系列是 reasoning 模型，响应格式为：
    <think>... 思考过程 ...</think>\n{json}
或被 markdown 围栏包裹：
    <think>...</think>\n```json\n{json}\n```

Pydantic / json.loads 不能直接解析这种带前缀的响应。本模块提供 strip_reasoning_artifacts()
统一处理：剥离 <think> 块、剥离 markdown 围栏、检测截断的思考块。
"""
from __future__ import annotations

import re


class ResponseTruncatedError(ValueError):
    """模型在思考阶段就用完了 max_tokens —— <think> 已开但未关闭。

    抛出此异常表示：本次调用不应被当作"解析失败"而被静默吞掉，调用方应该
    重试或增大 max_tokens。"""


_THINK_RE = re.compile(r"</think>", re.IGNORECASE)
_FENCE_HEAD_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_TAIL_RE = re.compile(r"\s*```$")


def strip_reasoning_artifacts(raw: str) -> str:
    """剥离 reasoning 模型响应中的 <think> 块和 markdown 围栏。

    Args:
        raw: 模型原始响应字符串

    Returns:
        清理后的字符串（应为合法 JSON 文本）

    Raises:
        ResponseTruncatedError: 当 <think> 开启但未闭合，说明响应在思考阶段被截断
    """
    if raw is None:
        return ""
    has_open = "<think>" in raw
    has_close = "</think>" in raw
    if has_open and not has_close:
        raise ResponseTruncatedError(
            "Response truncated mid-thinking: <think> opened but </think> missing. "
            "Increase max_tokens or retry."
        )
    s = raw
    if has_close:
        s = _THINK_RE.split(s, maxsplit=1)[1]
    s = s.strip()
    if s.startswith("```"):
        s = _FENCE_HEAD_RE.sub("", s)
        s = _FENCE_TAIL_RE.sub("", s)
    return s.strip()
