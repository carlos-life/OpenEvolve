"""Fitness evaluation of candidate solutions."""

from __future__ import annotations

import json
from typing import Any, List

from openevolve.models import Individual
from openevolve.sandbox import Sandbox


class Evaluator:
    """Evaluate an Individual's code against a set of test cases."""

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    def evaluate(
        self,
        individual: Individual,
        test_cases: List[dict[str, Any]],
        function_name: str,
    ) -> float:
        """Run *individual.code* against *test_cases* and return a fitness in [0, 1].

        Each test case dict should contain:
          - ``input``: a tuple of positional arguments
          - ``expected``: the expected return value
          - ``weight`` (optional): relative importance (default 1.0)

        Fitness is the weighted fraction of passing tests, with a tiny bonus
        for fast execution.
        """
        if not test_cases:
            return 0.0

        # Build the test harness
        test_code = self._build_test_script(test_cases, function_name)
        result = self.sandbox.execute(individual.code, test_code)

        if not result.success:
            return 0.0

        # Parse the structured output produced by the test harness
        try:
            outcomes = json.loads(result.output.strip().split("\n")[-1])
        except (json.JSONDecodeError, IndexError):
            return 0.0

        total_weight = 0.0
        weighted_pass = 0.0
        for tc, passed in zip(test_cases, outcomes):
            w = tc.get("weight", 1.0)
            total_weight += w
            if passed:
                weighted_pass += w

        if total_weight == 0:
            return 0.0

        fitness = weighted_pass / total_weight

        # Small bonus for fast execution (max 0.05 bonus), only when at least one test passes
        if fitness > 0:
            time_bonus = max(0.0, 0.05 * (1.0 - result.execution_time / self.sandbox.timeout_seconds))
            fitness = min(1.0, fitness + time_bonus)

        return fitness

    # ------------------------------------------------------------------

    @staticmethod
    def _build_test_script(
        test_cases: List[dict[str, Any]], function_name: str
    ) -> str:
        """Build a Python script that runs the candidate function on every
        test case and prints a JSON list of booleans."""
        lines = [
            "import json",
            "results = []",
        ]
        for i, tc in enumerate(test_cases):
            input_args = tc["input"]
            expected = tc["expected"]
            lines.append(f"try:")
            lines.append(f"    _out_{i} = {function_name}(*{input_args!r})")
            lines.append(f"    results.append(_out_{i} == {expected!r})")
            lines.append(f"except Exception:")
            lines.append(f"    results.append(False)")
        lines.append("print(json.dumps(results))")
        return "\n".join(lines)
