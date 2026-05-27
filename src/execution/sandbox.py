from __future__ import annotations

"""
ScholarMind - 安全代码执行沙箱 (Phase 7 lite 加固版)
====================================================

【重要】这是**个人项目的 lite 防护**，不是生产级沙箱。
       适用场景：本机跑 numpy / scipy / matplotlib 仿真代码。
       不适用：执行不可信第三方代码、给外部用户跑任意 Python。
       想要生产级隔离 → 跑容器 / nsjail / firejail。

核心能力（不变）：
  1. 子进程执行 Python 代码
  2. 超时控制
  3. 输出捕获（stdout + stderr + figure 文件）
  4. 工作目录隔离

Phase 7.1 加固：
  - 环境变量白名单（PATH / PYTHONPATH / TEMP / TMPDIR / SYSTEMROOT 等系统必需）；
    剥离所有 *_API_KEY / *_TOKEN / *_SECRET 与项目专属 env，避免子进程拿到主进程凭证。
  - 资源上限（POSIX）：CPU 60s / 地址空间 1GB / 文件创建 50MB；
    Windows 用 Job Object 占位（python 标准库不支持 setrlimit，写一行说明而非默默关掉）。
  - import 白名单（软）：执行前扫描 import 语句，超出 RECOMMENDED_IMPORTS 的发警告但仍跑。
  - 危险模式黑名单（保留）：注释明确"检测但不拦截"。
"""

import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ScholarMind.Sandbox")

# ============================================================
# 安全配置
# ============================================================

DANGEROUS_PATTERNS = [
    (r'\bos\.system\b', "os.system 禁止使用（命令注入风险）"),
    (r'\bos\.popen\b', "os.popen 禁止使用"),
    (r'\bos\.exec\b', "os.exec* 禁止使用"),
    (r'\bos\.remove\b', "os.remove 禁止使用（文件删除风险）"),
    (r'\bos\.rmdir\b', "os.rmdir 禁止使用"),
    (r'\bos\.unlink\b', "os.unlink 禁止使用"),
    (r'\bshutil\.rmtree\b', "shutil.rmtree 禁止使用"),
    (r'\bsubprocess\b', "subprocess 模块禁止使用"),
    (r'\b__import__\b', "__import__ 禁止使用"),
    (r'\beval\s*\(', "eval() 禁止使用"),
    (r'\bexec\s*\(', "exec() 禁止使用"),
    (r'\bopen\s*\(.*(w|a|x)', "文件写入限制（仅限输出目录）"),
]

# 学术常用，扫描 import 超出此集合时仅发警告（不拦截，保留学术灵活性）
RECOMMENDED_IMPORTS = {
    # 数值 / 科学
    "numpy", "np",
    "scipy", "scipy.signal", "scipy.linalg", "scipy.fft", "scipy.stats",
    "scipy.optimize", "scipy.integrate",
    "matplotlib", "matplotlib.pyplot", "plt", "matplotlib.figure",
    "pandas", "pd",
    "sympy",
    "math", "cmath", "statistics", "fractions", "decimal",
    "random",
    # 编程基础
    "json", "csv", "io", "re",
    "itertools", "functools", "collections",
    "dataclasses",
    "typing",
    "pathlib",
    # 字符串 / 时间
    "string", "datetime", "time",
    # 我们自己的 preamble 用
    "os", "sys",  # （扫到时不警告，但下游受白名单 env 约束）
}

# 沙箱子进程允许继承的环境变量（其余统一剥离）
_ENV_INHERIT_ALWAYS = {
    "PATH", "PYTHONPATH",
    "TEMP", "TMP", "TMPDIR", "TEMPDIR",
    "PYTHONIOENCODING",
}
# Windows 系统目录变量（缺它们 python.exe 启动会出问题）
_ENV_INHERIT_WINDOWS = {
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "USERPROFILE",
    "LOCALAPPDATA", "APPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)",
    "PROGRAMDATA", "COMSPEC", "PATHEXT", "OS",
}
# 永远剥离的敏感模式
_ENV_STRIP_PATTERNS = [
    re.compile(r".*_API_KEY$", re.IGNORECASE),
    re.compile(r".*_TOKEN$", re.IGNORECASE),
    re.compile(r".*_SECRET$", re.IGNORECASE),
    re.compile(r".*_PASSWORD$", re.IGNORECASE),
    re.compile(r"^MINIMAX_.*", re.IGNORECASE),
    re.compile(r"^NOTION_.*", re.IGNORECASE),
    re.compile(r"^OPENAI_.*", re.IGNORECASE),
    re.compile(r"^ANTHROPIC_.*", re.IGNORECASE),
    re.compile(r"^AWS_.*", re.IGNORECASE),
    re.compile(r"^GH(P|U|S|R|O)?_.*", re.IGNORECASE),
    re.compile(r"^SCHOLARMIND_.*", re.IGNORECASE),
]

DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 120
MAX_OUTPUT_LENGTH = 10000

# 资源上限（POSIX setrlimit）
RLIMIT_CPU_SECONDS = 60
RLIMIT_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024  # 1 GB
RLIMIT_FSIZE_BYTES = 50 * 1024 * 1024            # 50 MB

_IS_WINDOWS = sys.platform.startswith("win")


# ============================================================
# 数据类
# ============================================================


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time: float = 0.0
    output_files: list[str] = field(default_factory=list)
    error_message: str = ""
    security_warnings: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = []
        status = "✅ 执行成功" if self.success else "❌ 执行失败"
        lines.append(f"## {status}\n")
        lines.append(f"- ⏱️ 执行时间: {self.execution_time:.2f}s")
        lines.append(f"- 返回码: {self.return_code}")
        if self.security_warnings:
            lines.append(f"\n### ⚠️ 安全警告\n")
            for w in self.security_warnings:
                lines.append(f"- {w}")
        if self.stdout:
            output = self.stdout[:MAX_OUTPUT_LENGTH]
            if len(self.stdout) > MAX_OUTPUT_LENGTH:
                output += f"\n... (截断，共 {len(self.stdout)} 字符)"
            lines.append(f"\n### 📤 输出\n\n```\n{output}\n```")
        if self.stderr:
            lines.append(f"\n### ⚠️ 错误信息\n\n```\n{self.stderr[:3000]}\n```")
        if self.error_message:
            lines.append(f"\n### 💥 错误\n\n{self.error_message}")
        if self.output_files:
            lines.append(f"\n### 📁 生成的文件\n")
            for f in self.output_files:
                lines.append(f"- `{f}`")
        return "\n".join(lines)


# ============================================================
# 安全检查
# ============================================================


def check_code_safety(code: str) -> list[str]:
    """静态扫描危险模式（黑名单，仅警告不拦截）。"""
    warnings = []
    for pattern, message in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            warnings.append(message)
    return warnings


_IMPORT_FROM = re.compile(r"^\s*from\s+([\w\.]+)\s+import\b", re.MULTILINE)
_IMPORT_PLAIN = re.compile(r"^\s*import\s+([^\n#]+)", re.MULTILINE)


def _imported_modules(code: str) -> list[str]:
    """从源码里解析出 import 的模块名（去掉 `as` 别名 / 子模块、合并 `import a, b`）。"""
    mods: list[str] = []
    for m in _IMPORT_FROM.finditer(code):
        mods.append(m.group(1).strip())
    for m in _IMPORT_PLAIN.finditer(code):
        # 形如 "numpy as np, scipy.signal" → 拆 "," 后每个清 "as alias"
        for part in m.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            head = part.split(" as ")[0].strip()
            if head:
                mods.append(head)
    return mods


def check_import_whitelist(code: str) -> list[str]:
    """扫描 import 语句，超出 RECOMMENDED_IMPORTS 的模块发警告。

    实现策略：只看根模块名（"scipy.signal" → "scipy" 也算命中）。
    返回的是"软警告"，调用方决定是否阻塞（当前不阻塞）。
    """
    warnings: list[str] = []
    seen: set[str] = set()
    for token in _imported_modules(code):
        root = token.split(".")[0]
        if not root or root in seen:
            continue
        seen.add(root)
        if root in RECOMMENDED_IMPORTS or token in RECOMMENDED_IMPORTS:
            continue
        warnings.append(f"非白名单 import: `{token}` （沙箱将仍然执行，但请确认是否必要）")
    return warnings


# ============================================================
# 环境变量白名单
# ============================================================


def _build_sandbox_env(extra_overrides: dict[str, str] | None = None) -> dict[str, str]:
    """从 os.environ 派生子进程 env：只保留白名单 + 系统必需 + 强制 strip 模式。

    extra_overrides 在 strip 之后应用（用于 sandbox 自己设的 PYTHONDONTWRITEBYTECODE 等）。
    """
    inherit = set(_ENV_INHERIT_ALWAYS)
    if _IS_WINDOWS:
        inherit |= _ENV_INHERIT_WINDOWS

    env: dict[str, str] = {}
    for k, v in os.environ.items():
        ku = k.upper()
        # 1) strip 模式优先
        if any(p.match(k) for p in _ENV_STRIP_PATTERNS):
            continue
        # 2) 白名单
        if ku in {x.upper() for x in inherit}:
            env[k] = v

    if extra_overrides:
        env.update(extra_overrides)
    return env


# ============================================================
# 资源上限（POSIX preexec_fn）
# ============================================================


def _make_resource_limit_preexec():
    """构造 subprocess 的 preexec_fn，用于 POSIX setrlimit。

    Windows 不支持 setrlimit，返回 None（让 subprocess 用默认行为）。
    生产场景 Windows 应该用 Job Object，但本 lite 沙箱不引入额外依赖。
    """
    if _IS_WINDOWS:
        return None

    try:
        import resource  # type: ignore
    except ImportError:
        logger.debug("resource module unavailable; skipping setrlimit")
        return None

    def _set_limits():  # 在 fork 之后、exec 之前执行
        try:
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (RLIMIT_CPU_SECONDS, RLIMIT_CPU_SECONDS),
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (RLIMIT_ADDRESS_SPACE_BYTES, RLIMIT_ADDRESS_SPACE_BYTES),
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (RLIMIT_FSIZE_BYTES, RLIMIT_FSIZE_BYTES),
            )
        except Exception:  # noqa: BLE001
            pass

    return _set_limits


# ============================================================
# 沙箱执行引擎
# ============================================================


class CodeSandbox:
    """
    Lite 代码执行沙箱（Phase 7 加固版）。

    每次 execute() 在独立 tempdir 中运行；
    env 走白名单（不再透传 *_API_KEY 等敏感变量）；
    POSIX 上加 setrlimit CPU/内存/磁盘上限；
    import 与危险模式扫描仅作软警告。
    """

    def __init__(
        self,
        work_dir: Optional[Path] = None,
        timeout: int = DEFAULT_TIMEOUT,
        python_executable: Optional[str] = None,
    ):
        self.work_dir = work_dir or Path(tempfile.gettempdir()) / "scholarmind_sandbox"
        self.timeout = min(timeout, MAX_TIMEOUT)
        self.python_executable = python_executable or sys.executable
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
        skip_safety_check: bool = False,
    ) -> ExecutionResult:
        timeout = min(timeout or self.timeout, MAX_TIMEOUT)

        # Step 1: 安全检查（黑名单 + 白名单 import 都是软警告）
        security_warnings: list[str] = []
        if not skip_safety_check:
            security_warnings += check_code_safety(code)
            security_warnings += check_import_whitelist(code)
            if security_warnings:
                logger.warning("沙箱安全扫描 %d 条警告", len(security_warnings))

        # Step 2: 准备执行环境
        exec_dir = Path(tempfile.mkdtemp(dir=self.work_dir, prefix="exec_"))
        script_path = exec_dir / "main.py"

        preamble = (
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "import os\n"
            f"os.chdir(r'{exec_dir}')\n"
            "\n"
        )
        full_code = preamble + code
        full_code += (
            "\n\n"
            "# Auto-save all open figures\n"
            "import matplotlib.pyplot as _plt\n"
            "for _i, _fig in enumerate(_plt.get_fignums()):\n"
            "    _plt.figure(_fig)\n"
            f"    _plt.savefig(r'{exec_dir}' + f'/figure_{{_i}}.png', dpi=150, bbox_inches='tight')\n"
            "_plt.close('all')\n"
        )
        script_path.write_text(full_code, encoding="utf-8")

        # Step 3: 执行（env 白名单 + 资源上限）
        sandbox_env = _build_sandbox_env({
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",
            "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING", "utf-8"),
        })
        preexec = _make_resource_limit_preexec()

        start_time = time.time()
        try:
            kwargs = {
                "capture_output": True,
                "text": True,
                "timeout": timeout,
                "cwd": str(exec_dir),
                "env": sandbox_env,
            }
            if preexec is not None:
                kwargs["preexec_fn"] = preexec

            result = subprocess.run(
                [self.python_executable, str(script_path)],
                **kwargs,
            )
            execution_time = time.time() - start_time

            output_files = [
                str(f) for f in exec_dir.iterdir()
                if f.suffix in (".png", ".jpg", ".csv", ".json", ".txt", ".pdf")
                and f.name != "main.py"
            ]

            return ExecutionResult(
                success=(result.returncode == 0),
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                execution_time=execution_time,
                output_files=output_files,
                security_warnings=security_warnings,
            )

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                return_code=-1,
                execution_time=execution_time,
                error_message=f"⏰ 执行超时（{timeout}s）。可能存在无限循环或计算量过大。",
                security_warnings=security_warnings,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                return_code=-2,
                execution_time=execution_time,
                error_message=f"执行异常: {type(e).__name__}: {str(e)}",
                security_warnings=security_warnings,
            )
