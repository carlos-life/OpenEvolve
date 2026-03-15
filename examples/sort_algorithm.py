"""Example: Evolve a sorting algorithm."""

import asyncio

from openevolve import Evolution, EvolutionConfig


def main():
    config = EvolutionConfig(
        population_size=10,
        generations=5,
        mutation_rate=0.3,
        elite_ratio=0.2,
        tournament_size=3,
        timeout_seconds=10,
    )

    engine = Evolution(config)

    problem = (
        "Write a Python function that sorts a list of integers in ascending order. "
        "Be creative - try to find an efficient approach."
    )
    signature = "def solution(arr: list) -> list:"
    test_cases = [
        {"input": ([3, 1, 2],), "expected": [1, 2, 3]},
        {"input": ([],), "expected": []},
        {"input": ([1],), "expected": [1]},
        {"input": ([5, 4, 3, 2, 1],), "expected": [1, 2, 3, 4, 5]},
        {"input": ([1, 1, 1],), "expected": [1, 1, 1]},
        {"input": ([2, 3, 1, 4, 5],), "expected": [1, 2, 3, 4, 5]},
        {"input": ([-1, 0, 1],), "expected": [-1, 0, 1]},
        {"input": ([100, -100],), "expected": [-100, 100]},
    ]

    def on_generation(gen: int, best_fitness: float, avg_fitness: float):
        print(f"  Generation {gen}: best={best_fitness:.3f}  avg={avg_fitness:.3f}")

    print("Evolving a sorting algorithm...")
    result = asyncio.run(
        engine.run(
            problem_description=problem,
            function_signature=signature,
            test_cases=test_cases,
            function_name="solution",
            on_generation=on_generation,
        )
    )

    print(f"\nEvolution complete!")
    print(f"  Best fitness: {result.best_individual.fitness:.3f}")
    print(f"  Total evaluations: {result.total_evaluations}")
    print(f"  Elapsed: {result.elapsed_seconds:.1f}s")
    print(f"\nBest code:\n{result.best_individual.code}")


if __name__ == "__main__":
    main()
