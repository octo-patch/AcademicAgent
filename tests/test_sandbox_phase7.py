"""Phase 7 验证：沙箱 lite 加固（env 白名单 + setrlimit + import 白名单警告）。"""

from __future__ import annotations

import os
import sys

import pytest

from src.execution.sandbox import (
    CodeSandbox,
    _build_sandbox_env,
    check_import_whitelist,
)


# ============================================================
# env 白名单
# ============================================================


def test_build_sandbox_env_strips_api_keys(monkeypatch):
    """MINIMAX_API_KEY / NOTION_API_KEY 等敏感变量必须被剥离。"""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-secret-do-not-leak")
    monkeypatch.setenv("NOTION_API_KEY", "ntn-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-other")
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
    monkeypatch.setenv("MY_DB_PASSWORD", "p@ss")
    monkeypatch.setenv("SCHOLARMIND_DATA_DIR", "/tmp/x")

    env = _build_sandbox_env()
    sensitive = [
        "MINIMAX_API_KEY", "NOTION_API_KEY", "OPENAI_API_KEY",
        "GITHUB_TOKEN", "MY_DB_PASSWORD", "SCHOLARMIND_DATA_DIR",
    ]
    for k in sensitive:
        assert k not in env, f"敏感变量 {k} 不应出现在沙箱 env 里"


def test_build_sandbox_env_keeps_path():
    env = _build_sandbox_env()
    # PATH 是绝对必需的（subprocess 找 python.exe）
    assert "PATH" in env, "PATH 应该在白名单里"


def test_build_sandbox_env_applies_overrides(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "secret")
    env = _build_sandbox_env({"MPLBACKEND": "Agg", "MY_FORCED": "yes"})
    assert env.get("MPLBACKEND") == "Agg"
    assert env.get("MY_FORCED") == "yes"
    assert "MINIMAX_API_KEY" not in env


def test_subprocess_cannot_see_api_key(monkeypatch, tmp_path):
    """端到端：在 monkeypatch 设了 MINIMAX_API_KEY 后跑沙箱代码，
    沙箱里 os.getenv('MINIMAX_API_KEY') 应该是 None。"""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-must-not-leak-123")
    sb = CodeSandbox(work_dir=tmp_path)
    code = (
        "import os\n"
        "print('MINIMAX_API_KEY =', repr(os.environ.get('MINIMAX_API_KEY')))\n"
        "print('NOTION_API_KEY =', repr(os.environ.get('NOTION_API_KEY')))\n"
    )
    result = sb.execute(code, timeout=10)
    assert result.success, f"sandbox 执行失败: {result.stderr}\n{result.error_message}"
    # 子进程里应看不到 key（None 字符串化）
    assert "sk-must-not-leak-123" not in result.stdout
    assert "MINIMAX_API_KEY = None" in result.stdout


# ============================================================
# import 白名单
# ============================================================


def test_import_whitelist_allows_numpy_scipy_matplotlib():
    code = (
        "import numpy as np\n"
        "from scipy.signal import butter\n"
        "import matplotlib.pyplot as plt\n"
    )
    assert check_import_whitelist(code) == []


def test_import_whitelist_flags_unknown_module():
    code = "import socket\nimport some_obscure_lib\n"
    warns = check_import_whitelist(code)
    assert any("socket" in w for w in warns)
    assert any("some_obscure_lib" in w for w in warns)


def test_import_whitelist_does_not_block_execution(tmp_path):
    """import 白名单是"软警告"：超出白名单的代码仍能执行。"""
    sb = CodeSandbox(work_dir=tmp_path)
    code = "import socket\nprint('socket ok')\n"
    result = sb.execute(code, timeout=10)
    assert result.success
    assert "socket ok" in result.stdout
    # 但应有警告
    assert any("socket" in w for w in result.security_warnings)


# ============================================================
# 资源上限（POSIX）
# ============================================================


@pytest.mark.skipif(sys.platform.startswith("win"), reason="setrlimit POSIX only")
def test_setrlimit_memory_kills_large_allocation(tmp_path):
    """大内存分配应被 setrlimit RLIMIT_AS 杀掉（仅 POSIX）。"""
    sb = CodeSandbox(work_dir=tmp_path)
    # 尝试分配 2GB（超 RLIMIT_AS 1GB）—— 应失败
    code = (
        "x = bytearray(2 * 1024 * 1024 * 1024)\n"
        "print('allocated', len(x))\n"
    )
    result = sb.execute(code, timeout=20)
    # MemoryError 或非零返回码即视为防护生效
    assert (not result.success) or ("MemoryError" in (result.stderr or ""))


# ============================================================
# 工作目录 / 子进程隔离不变
# ============================================================


def test_sandbox_cwd_is_exec_dir(tmp_path):
    """子进程 cwd 应在 exec_<x>/，不是主进程 cwd。"""
    sb = CodeSandbox(work_dir=tmp_path)
    result = sb.execute("import os; print(os.getcwd())", timeout=10)
    assert result.success
    assert "exec_" in result.stdout
    assert str(tmp_path) in result.stdout
