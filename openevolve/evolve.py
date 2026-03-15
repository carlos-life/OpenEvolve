"""Main evolution engine."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Callable, List, Optional

from openevolve.evaluator import Evaluator
from openevolve.llm import LLMClient, MockLLMClient
from openevolve.models import EvolutionConfig, EvolutionResult, Individual
from openevolve.population import Population
from openevolve.sandbox import Sandbox


class Evolution:
    """Orchestrates the evolutionary process."""

    def __init__(self, config: EvolutionConfig, llm_client: Optional[Any] = None):
        self.config = config
        self.sandbox = Sandbox(timeout_seconds=config.timeout_seconds)
        self.evaluator = Evaluator(self.sandbox)
        if llm_client is not None:
            self.llm = llm_client
        elif config.llm_api_key:
            base_url = config.llm_base_url or "https://api.openai.com/v1"
            self.llm = LLMClient(
                model=config.llm_model,
                api_key=config.llm_api_key,
                base_url=base_url,
            )
        else:
            self.llm = MockLLMClient()

    # ------------------------------------------------------------------

    async def run(
        self,
        problem_description: str,
        function_signature: str,
        test_cases: List[dict[str, Any]],
        function_name: str = "solution",
        initial_code: str = "",
        on_generation: Optional[Callable] = None,
    ) -> EvolutionResult:
        """Run the full evolutionary optimisation loop.

        Parameters
        ----------
        problem_description:
            Natural-language description of the problem.
        function_signature:
            Python signature, e.g. ``"def solution(arr: list) -> list:"``.
        test_cases:
            List of dicts with keys ``input``, ``expected``, and optional
            ``weight``.
        function_name:
            Name of the function inside the candidate code.
        initial_code:
            Optional seed code.  If provided it is added as the first
            individual and fewer LLM-generated seeds are created.
        on_generation:
            Optional callback ``(gen, best_fitness, avg_fitness) -> None``.
        """
        start_time = time.monotonic()
        population = Population(self.config)
        total_evals = 0

        # ----- 1. Initial population -----------------------------------
        if initial_code:
            ind = Individual(code=initial_code, generation=0)
            ind.fitness = self.evaluator.evaluate(ind, test_cases, function_name)
            population.add(ind)
            total_evals += 1
            n_generate = max(0, self.config.population_size - 1)
        else:
            n_generate = self.config.population_size

        if n_generate > 0:
            codes = await self.llm.generate_initial(
                problem_description, function_signature, n_generate
            )
            for code in codes:
                ind = Individual(code=code, generation=0)
                ind.fitness = self.evaluator.evaluate(ind, test_cases, function_name)
                population.add(ind)
                total_evals += 1

        # Record generation-0 stats
        gen_stats: list[dict[str, Any]] = [population.get_stats()]
        if on_generation:
            s = gen_stats[-1]
            _maybe_await(on_generation, 0, s["best_fitness"], s["avg_fitness"])

        # ----- 2. Generational loop ------------------------------------
        for gen in range(1, self.config.generations + 1):
            new_individuals: list[Individual] = []

            # Elitism — carry forward the best individuals unchanged
            elites = population.get_elite()
            for elite in elites:
                clone = Individual(
                    code=elite.code,
                    fitness=elite.fitness,
                    generation=gen,
                    parent_ids=[elite.id],
                    metadata={"origin": "elite"},
                )
                new_individuals.append(clone)

            # Fill the rest of the population
            while len(new_individuals) < self.config.population_size:
                if random.random() < self.config.mutation_rate:
                    # Mutation
                    parent = population.tournament_select()
                    child_code = await self.llm.mutate(
                        parent.code, parent.fitness, problem_description
                    )
                    child = Individual(
                        code=child_code,
                        generation=gen,
                        parent_ids=[parent.id],
                        metadata={"origin": "mutation"},
                    )
                else:
                    # Crossover
                    p1 = population.tournament_select()
                    p2 = population.tournament_select()
                    child_code = await self.llm.crossover(
                        p1.code, p1.fitness, p2.code, p2.fitness, problem_description
                    )
                    child = Individual(
                        code=child_code,
                        generation=gen,
                        parent_ids=[p1.id, p2.id],
                        metadata={"origin": "crossover"},
                    )
                child.fitness = self.evaluator.evaluate(child, test_cases, function_name)
                total_evals += 1
                new_individuals.append(child)

            # Replace population with new generation
            for ind in new_individuals:
                population.add(ind)

            stats = population.get_stats()
            gen_stats.append(stats)
            if on_generation:
                _maybe_await(on_generation, gen, stats["best_fitness"], stats["avg_fitness"])

        elapsed = time.monotonic() - start_time
        return EvolutionResult(
            best_individual=population.get_best(),
            generation_stats=gen_stats,
            total_evaluations=total_evals,
            elapsed_seconds=elapsed,
        )


def _maybe_await(fn: Callable, *args: Any) -> None:
    """Call *fn* — if it returns a coroutine, schedule it."""
    result = fn(*args)
    if asyncio.iscoroutine(result):
        asyncio.ensure_future(result)


# -- CLI entry point ---------------------------------------------------

def main() -> None:
    """Minimal CLI entry point."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="OpenEvolve — evolve algorithms with LLMs")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    config = EvolutionConfig(
        generations=args.generations,
        population_size=args.population_size,
        llm_model=args.model,
    )
    engine = Evolution(config)

    # Default demo: sort
    test_cases = [
        {"input": ([3, 1, 2],), "expected": [1, 2, 3]},
        {"input": ([],), "expected": []},
        {"input": ([1],), "expected": [1]},
        {"input": ([5, 4, 3, 2, 1],), "expected": [1, 2, 3, 4, 5]},
    ]

    def _on_gen(gen: int, best: float, avg: float) -> None:
        print(f"Generation {gen}: best={best:.3f}  avg={avg:.3f}")

    result = asyncio.run(
        engine.run(
            problem_description="Sort a list of integers in ascending order.",
            function_signature="def solution(arr: list) -> list:",
            test_cases=test_cases,
            on_generation=_on_gen,
        )
    )
    print(f"\nBest fitness: {result.best_individual.fitness:.3f}")
    print(f"Total evaluations: {result.total_evaluations}")
    print(f"Elapsed: {result.elapsed_seconds:.1f}s")
    print(f"\nBest code:\n{result.best_individual.code}")


if __name__ == "__main__":
    main()
