"""Tests for the Sandbox."""

import sys
import pytest

from openevolve.sandbox import Sandbox


class TestSandbox:
    def setup_method(self):
        self.sandbox = Sandbox(timeout_seconds=5)

    def test_successful_execution(self):
        code = "def add(a, b):\n    return a + b"
        test_code = "print(add(2, 3))"
        result = self.sandbox.execute(code, test_code)
        assert result.success
        assert result.output.strip() == "5"
        assert result.error == "" or result.error.strip() == ""
        assert result.execution_time > 0

    def test_syntax_error(self):
        code = "def bad(:\n    pass"
        test_code = "bad()"
        result = self.sandbox.execute(code, test_code)
        assert not result.success
        assert result.error  # should contain syntax error info

    def test_runtime_error(self):
        code = "def oops():\n    return 1 / 0"
        test_code = "oops()"
        result = self.sandbox.execute(code, test_code)
        assert not result.success
        assert "ZeroDivision" in result.error

    def test_timeout(self):
        sandbox = Sandbox(timeout_seconds=2)
        code = "import time\ndef slow():\n    time.sleep(100)"
        test_code = "slow()"
        result = sandbox.execute(code, test_code)
        assert not result.success
        assert "timed out" in result.error.lower()

    def test_empty_output(self):
        code = "x = 42"
        test_code = ""
        result = self.sandbox.execute(code, test_code)
        assert result.success
        assert result.output.strip() == ""

    def test_stderr_capture(self):
        code = "import sys\nsys.stderr.write('warn\\n')"
        test_code = ""
        result = self.sandbox.execute(code, test_code)
        assert result.success
        assert "warn" in result.error

    def test_restricted_env(self):
        # Should not have access to user env vars
        code = "import os\nprint(os.environ.get('USER', 'NONE'))"
        test_code = ""
        result = self.sandbox.execute(code, test_code)
        assert result.success
        assert result.output.strip() == "NONE"

    def test_multiline_output(self):
        code = "def greet(name):\n    return f'Hello, {name}!'"
        test_code = "print(greet('A'))\nprint(greet('B'))"
        result = self.sandbox.execute(code, test_code)
        assert result.success
        lines = result.output.strip().split("\n")
        assert lines == ["Hello, A!", "Hello, B!"]
