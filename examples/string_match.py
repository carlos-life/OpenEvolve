"""Example: Evolve a fuzzy string matching function."""

import asyncio

from openevolve import Evolution, EvolutionConfig


def main():
    config = EvolutionConfig(
        population_size=8,
        generations=5,
        mutation_rate=0.4,
        elite_ratio=0.2,
        tournament_size=3,
        timeout_seconds=10,
    )

    engine = Evolution(config)

    problem = (
        "Write a Python function that takes two strings and returns a float "
        "between 0.0 and 1.0 indicating how similar they are. "
        "1.0 means identical, 0.0 means completely different. "
        "Handle case insensitivity."
    )
    signature = "def solution(a: str, b: str) -> float:"
    test_cases = [
        {"input": ("hello", "hello"), "expected": 1.0},
        {"input": ("Hello", "hello"), "expected": 1.0},
        {"input": ("abc", "xyz"), "expected": 0.0},
        {"input": ("", ""), "expected": 1.0},
        {"input": ("ab", "abc"), "expected": True},  # just check it returns >0
    ]

    def on_generation(gen: int, best_fitness: float, avg_fitness: float):
        print(f"  Generation {gen}: best={best_fitness:.3f}  avg={avg_fitness:.3f}")

    print("Evolving a string matching function...")
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
