"""Tests for the Evolution engine using MockLLMClient."""

import asyncio
import pytest

from openevolve.evolve import Evolution
from openevolve.llm import MockLLMClient
from openevolve.models import EvolutionConfig


class TestEvolution:
    def _make_engine(self, **overrides):
        defaults = dict(
            population_size=6,
            generations=3,
            mutation_rate=0.3,
            elite_ratio=0.2,
            tournament_size=2,
            timeout_seconds=5,
        )
        defaults.update(overrides)
        config = EvolutionConfig(**defaults)
        mock_llm = MockLLMClient()
        return Evolution(config, llm_client=mock_llm)

    def test_full_evolution_sort(self):
        engine = self._make_engine()
        test_cases = [
            {"input": ([3, 1, 2],), "expected": [1, 2, 3]},
            {"input": ([],), "expected": []},
            {"input": ([1],), "expected": [1]},
            {"input": ([5, 4, 3, 2, 1],), "expected": [1, 2, 3, 4, 5]},
        ]
        result = asyncio.run(
            engine.run(
                problem_description="Sort a list of integers in ascending order.",
                function_signature="def solution(arr: list) -> list:",
                test_cases=test_cases,
                function_name="solution",
            )
        )
        assert result.best_individual is not None
        assert result.best_individual.fitness > 0.5
        assert result.total_evaluations > 0
        assert result.elapsed_seconds > 0
        assert len(result.generation_stats) == 4  # gen 0 + 3 generations

    def test_evolution_with_initial_code(self):
        engine = self._make_engine(generations=2, population_size=4)
        test_cases = [
            {"input": ([3, 1, 2],), "expected": [1, 2, 3]},
            {"input": ([],), "expected": []},
        ]
        initial = "def solution(arr: list) -> list:\n    return sorted(arr)"
        result = asyncio.run(
            engine.run(
                problem_description="Sort a list.",
                function_signature="def solution(arr: list) -> list:",
                test_cases=test_cases,
                function_name="solution",
                initial_code=initial,
            )
        )
        assert result.best_individual.fitness > 0.9

    def test_on_generation_callback(self):
        engine = self._make_engine(generations=2, population_size=4)
        test_cases = [
            {"input": ([3, 1, 2],), "expected": [1, 2, 3]},
        ]
        callback_log = []

        def on_gen(gen, best, avg):
            callback_log.append((gen, best, avg))

        asyncio.run(
            engine.run(
                problem_description="Sort a list of integers.",
                function_signature="def solution(arr: list) -> list:",
                test_cases=test_cases,
                function_name="solution",
                on_generation=on_gen,
            )
        )
        # Should have been called for gen 0, 1, 2
        assert len(callback_log) == 3
        for gen, best, avg in callback_log:
            assert isinstance(gen, int)
            assert isinstance(best, float)
            assert isinstance(avg, float)

    def test_generation_stats_structure(self):
        engine = self._make_engine(generations=2, population_size=4)
        test_cases = [{"input": ([2, 1],), "expected": [1, 2]}]
        result = asyncio.run(
            engine.run(
                problem_description="Sort integers.",
                function_signature="def solution(arr: list) -> list:",
                test_cases=test_cases,
                function_name="solution",
            )
        )
        for stats in result.generation_stats:
            assert "size" in stats
            assert "avg_fitness" in stats
            assert "best_fitness" in stats
            assert "diversity" in stats

    def test_best_individual_has_code(self):
        engine = self._make_engine(generations=1, population_size=4)
        test_cases = [{"input": ([1],), "expected": [1]}]
        result = asyncio.run(
            engine.run(
                problem_description="Sort a list.",
                function_signature="def solution(arr: list) -> list:",
                test_cases=test_cases,
                function_name="solution",
            )
        )
        assert result.best_individual.code.strip() != ""
        assert "def solution" in result.best_individual.code

    def test_no_api_key_uses_mock(self):
        config = EvolutionConfig(population_size=4, generations=1)
        engine = Evolution(config)
        # Should auto-use MockLLMClient since no api_key
        assert isinstance(engine.llm, MockLLMClient)
