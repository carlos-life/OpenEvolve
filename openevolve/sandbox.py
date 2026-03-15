"""Sandboxed code execution."""

from __future__ import annotations

import subprocess
import tempfile
import time
import os
from pathlib import Path

from openevolve.models import SandboxResult


class Sandbox:
    """Execute untrusted code in a subprocess with timeout enforcement."""

    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds

    def execute(self, code: str, test_code: str) -> SandboxResult:
        """Execute *code* followed by *test_code* in an isolated subprocess.

        The combined script is written to a temporary file and run with the
        current Python interpreter.  stdout, stderr, and the return code are
        captured.  A hard timeout is enforced via ``subprocess.run``.
        """
        combined = code + "\n\n" + test_code
        tmp_dir = tempfile.mkdtemp(prefix="openevolve_")
        script_path = Path(tmp_dir) / "script.py"
        script_path.write_text(combined)

        start = time.monotonic()
        try:
            result = subprocess.run(
                [os.sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=tmp_dir,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "HOME": tmp_dir,
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
            elapsed = time.monotonic() - start
            return SandboxResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                execution_time=elapsed,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            return SandboxResult(
                success=False,
                output="",
                error="Execution timed out",
                execution_time=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return SandboxResult(
                success=False,
                output="",
                error=str(exc),
                execution_time=elapsed,
            )
        finally:
            # Clean up temp files
            try:
                script_path.unlink(missing_ok=True)
                Path(tmp_dir).rmdir()
            except OSError:
                pass
