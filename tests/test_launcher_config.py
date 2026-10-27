"""Shell launchers preserve caller overrides without executing real services."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")


@pytest.mark.parametrize("script", ["run_query.sh", "run_ingest.sh", "serve_api.sh"])
def test_production_launchers_preserve_caller_environment(script, tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / ".env").write_text("RECALL_TOP_K=77\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("RECALL_TOP_K=40\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    shutil.copy(root / "scripts" / script, tmp_path / "scripts" / script)
    shutil.copy(root / "scripts" / "env_defaults.sh", tmp_path / "scripts" / "env_defaults.sh")
    (tmp_path / "python").write_text("#!/bin/sh\nprintf '%s' \"$RECALL_TOP_K\"\n", encoding="utf-8")
    (tmp_path / "uvicorn").write_text("#!/bin/sh\nprintf '%s' \"$RECALL_TOP_K\"\n", encoding="utf-8")
    (tmp_path / "python").chmod(0o755)
    (tmp_path / "uvicorn").chmod(0o755)

    result = subprocess.run(
        ["bash", str(tmp_path / "scripts" / script)],
        env=dict(
            os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"], RECALL_TOP_K="99", PYTHON=sys.executable
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "99"
