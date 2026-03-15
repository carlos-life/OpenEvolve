"""Tests for the Evaluator."""

import pytest

from openevolve.evaluator import Evaluator
from openevolve.models import Individual
from openevolve.sandbox import Sandbox


class TestEvaluator:
    def setup_method(self):
        self.sandbox = Sandbox(timeout_seconds=5)
        self.evaluator = Evaluator(self.sandbox)

    def test_perfect_score(self):
        ind = Individual(code="def solution(arr: list) -> list:\n    return sorted(arr)")
        test_cases = [
            {"input": ([3, 1, 2],), "expected": [1, 2, 3]},
            {"input": ([],), "expected": []},
            {"input": ([1],), "expected": [1]},
        ]
        fitness = self.evaluator.evaluate(ind, test_cases, "solution")
        # Should be very close to 1.0 (1.0 + tiny time bonus, capped at 1.0)
        assert fitness > 0.9

    def test_zero_score_bad_code(self):
        ind = Individual(code="def solution(arr):\n    return 'wrong'")
        test_cases = [
            {"input": ([3, 1, 2],), "expected": [1, 2, 3]},
        ]
        fitness = self.evaluator.evaluate(ind, test_cases, "solution")
        assert fitness == 0.0

    def test_partial_score(self):
        # This function only works for empty and single-element lists
        ind = Individual(
            code=(
                "def solution(arr: list) -> list:\n"
                "    return list(arr)"
            )
        )
        test_cases = [
            {"input": ([3, 1, 2],), "expected": [1, 2, 3]},
            {"input": ([],), "expected": []},
            {"input": ([1],), "expected": [1]},
            {"input": ([2, 1],), "expected": [1, 2]},
        ]
        fitness = self.evaluator.evaluate(ind, test_cases, "solution")
        # 2 out of 4 pass (empty, single-element)
        assert 0.4 <= fitness <= 0.6

    def test_weighted_test_cases(self):
        ind = Individual(code="def solution(x):\n    return x * 2")
        test_cases = [
            {"input": (2,), "expected": 4, "weight": 10.0},   # passes
            {"input": (3,), "expected": 7, "weight": 1.0},    # fails (returns 6)
        ]
        fitness = self.evaluator.evaluate(ind, test_cases, "solution")
        # 10/(10+1) ~ 0.909
        assert fitness > 0.85

    def test_empty_test_cases(self):
        ind = Individual(code="def solution(): pass")
        fitness = self.evaluator.evaluate(ind, [], "solution")
        assert fitness == 0.0

    def test_syntax_error_code(self):
        ind = Individual(code="def solution(arr):\n    return [[[")
        test_cases = [{"input": ([1],), "expected": [1]}]
        fitness = self.evaluator.evaluate(ind, test_cases, "solution")
        assert fitness == 0.0

    def test_exception_in_candidate(self):
        ind = Individual(code="def solution(arr):\n    raise ValueError('nope')")
        test_cases = [{"input": ([1],), "expected": [1]}]
        fitness = self.evaluator.evaluate(ind, test_cases, "solution")
        assert fitness == 0.0
