"""轻量 ``.env`` 加载器（零依赖）。

背景
----
``scripts/*.sh`` 通过 ``set -a; source .env; set +a`` 把 ``.env`` 注入环境变量，
但下面这些**直接运行**的方式不会加载 ``.env``：

- ``python -m pcb_rag.ingest`` / ``python -m pcb_rag.query``
- ``uvicorn pcb_rag.dify_external_api:app``
- ``python eval/evaluate_recall.py`` / ``python eval/evaluate.py``

结果是「明明在 ``.env`` 里填了 API Key，却报缺少配置」。本模块提供一个最小实现，
避免为了这一个功能引入 ``python-dotenv`` 依赖。

约定
----
- 只写入**尚不存在**的环境变量（``os.environ.setdefault`` 语义）：
  已经 ``export`` 的真实环境变量、以及 shell 脚本 ``source`` 过的值都不会被覆盖，
  因此与现有启动脚本的加载顺序完全兼容。
- 支持 ``KEY=VALUE``、``export KEY=VALUE``、``#`` 整行注释、值外层单/双引号，
  以及 ``VALUE  # 行尾注释``。
- 找不到 ``.env`` 时静默返回 0，不抛异常（测试与 CI 环境通常没有 ``.env``）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

__all__ = ["load_project_env", "project_root"]


def project_root() -> Path:
    """返回仓库根目录（本文件位于 ``<root>/src/pcb_rag/``）。"""

    return Path(__file__).resolve().parents[2]


def _unquote(value: str) -> str:
    """去掉值外层的引号；未加引号时剥离 `` #`` 之后的行尾注释。"""

    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] in {"'", '"'} and stripped[-1] == stripped[0]:
        return stripped[1:-1]
    return stripped.split(" #", 1)[0].strip()


def load_project_env(path: Optional[Union[str, Path]] = None) -> int:
    """把 ``.env`` 载入 ``os.environ``，返回**新写入**的键数量。

    Args:
        path: 显式指定的 ``.env`` 路径；默认使用仓库根目录下的 ``.env``。
    """

    env_path = Path(path) if path is not None else project_root() / ".env"
    if not env_path.is_file():
        return 0

    written = 0
    try:
        content = env_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        os.environ[key] = _unquote(value)
        written += 1

    return written
