"""Population management with selection and elitism."""

from __future__ import annotations

import random
from typing import List

from openevolve.models import EvolutionConfig, Individual


class Population:
    """Manages a collection of Individuals across generations."""

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self._individuals: list[Individual] = []

    # -- mutators -------------------------------------------------------

    def add(self, individual: Individual) -> None:
        """Add an individual to the population."""
        self._individuals.append(individual)

    # -- queries --------------------------------------------------------

    def get_elite(self) -> List[Individual]:
        """Return the top *elite_ratio* fraction of the current population,
        sorted by descending fitness."""
        if not self._individuals:
            return []
        n = max(1, int(len(self._individuals) * self.config.elite_ratio))
        return sorted(self._individuals, key=lambda ind: ind.fitness, reverse=True)[:n]

    def tournament_select(self) -> Individual:
        """Select an individual via tournament selection."""
        if not self._individuals:
            raise ValueError("Population is empty")
        size = min(self.config.tournament_size, len(self._individuals))
        competitors = random.sample(self._individuals, size)
        return max(competitors, key=lambda ind: ind.fitness)

    def get_best(self) -> Individual:
        """Return the individual with the highest fitness."""
        if not self._individuals:
            raise ValueError("Population is empty")
        return max(self._individuals, key=lambda ind: ind.fitness)

    def get_generation(self, gen: int) -> List[Individual]:
        """Return all individuals belonging to a specific generation."""
        return [ind for ind in self._individuals if ind.generation == gen]

    def get_stats(self) -> dict:
        """Compute summary statistics for the population."""
        if not self._individuals:
            return {"size": 0, "avg_fitness": 0.0, "best_fitness": 0.0, "diversity": 0.0}
        fitnesses = [ind.fitness for ind in self._individuals]
        unique_codes = len({ind.code for ind in self._individuals})
        return {
            "size": len(self._individuals),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "best_fitness": max(fitnesses),
            "diversity": unique_codes / len(self._individuals) if self._individuals else 0.0,
        }

    @property
    def size(self) -> int:
        return len(self._individuals)

    @property
    def individuals(self) -> list[Individual]:
        return list(self._individuals)
