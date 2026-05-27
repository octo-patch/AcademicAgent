"""
mcp_servers 内部兼容层
=========================

让 ``src/mcp_servers/*.py`` 既能作为真正的 MCP server 运行（需要 ``mcp[cli]``
依赖、Python 3.10+），也能在缺 mcp 包的环境下被 ``src.runtime.tools`` 等
Python 内部调用方按普通模块导入。

为什么：runtime/tools.py 明确把 mcp_servers 里的 ``@mcp.tool()`` async
函数作为「内部 Python 函数」复用（见该文件顶部注释「runtime 跑在同一个
Python 进程内，直接调内部函数省 stdio 一跳」）。如果 mcp_servers 的顶部
``from mcp.server.fastmcp import FastMCP`` 是硬导入，缺 mcp 的环境会让
runtime workflow 整条链路无法启动。

用法（在每个 server 文件里）：

    from ._mcp_compat import make_fastmcp_or_stub
    mcp = make_fastmcp_or_stub(
        "ScholarMind-PaperSearch",
        instructions="...",
    )

- 装有 ``mcp[cli]`` 时返回真实 ``FastMCP``，行为完全一致；
- 缺包时返回轻量 stub：``@mcp.tool()`` 装饰器透传函数，``mcp.run()`` 抛
  明确的 ``RuntimeError`` 告诉调用者「该作为 server 跑就装上 mcp」。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("ScholarMind.MCPCompat")


class _FastMCPStub:
    """轻量替身。装饰器透明，跑 server 时显式报错。"""

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self._kwargs = kwargs

    def tool(self, *_args, **_kwargs):
        def _decorator(fn):
            return fn
        return _decorator

    def resource(self, *_args, **_kwargs):
        def _decorator(fn):
            return fn
        return _decorator

    def prompt(self, *_args, **_kwargs):
        def _decorator(fn):
            return fn
        return _decorator

    def run(self, *_args, **_kwargs):
        raise RuntimeError(
            f"MCP server {self.name!r} 不可用：mcp 包未安装。"
            " 若需作为 MCP server 运行，请 pip install 'mcp[cli]>=1.0.0' (需要 Python>=3.10)。"
        )


def make_fastmcp_or_stub(name: str, **kwargs: Any):
    """优先返回真实 FastMCP；ImportError 时回退到 stub。"""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        logger.info(
            "mcp 包未安装，%s 走 stub（内部 Python 调用不受影响；不能作为 MCP server 启动）",
            name,
        )
        return _FastMCPStub(name, **kwargs)
    return FastMCP(name, **kwargs)
