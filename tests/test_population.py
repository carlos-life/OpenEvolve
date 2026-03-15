"""Tests for Population management."""

import pytest

from openevolve.models import EvolutionConfig, Individual
from openevolve.population import Population


class TestPopulation:
    def setup_method(self):
        self.config = EvolutionConfig(
            population_size=10,
            elite_ratio=0.2,
            tournament_size=3,
        )
        self.pop = Population(self.config)

    def _add_individuals(self, n: int, fitness_fn=None):
        for i in range(n):
            f = fitness_fn(i) if fitness_fn else float(i) / max(n - 1, 1)
            self.pop.add(Individual(code=f"code_{i}", fitness=f, generation=0))

    def test_add_and_size(self):
        assert self.pop.size == 0
        self._add_individuals(5)
        assert self.pop.size == 5

    def test_get_best(self):
        self._add_individuals(5)
        best = self.pop.get_best()
        assert best.fitness == 1.0

    def test_get_best_empty_raises(self):
        with pytest.raises(ValueError):
            self.pop.get_best()

    def test_get_elite(self):
        self._add_individuals(10)
        elites = self.pop.get_elite()
        # 20% of 10 = 2
        assert len(elites) == 2
        assert all(e.fitness >= 0.8 for e in elites)

    def test_get_elite_empty(self):
        assert self.pop.get_elite() == []

    def test_tournament_select(self):
        self._add_individuals(10)
        selected = self.pop.tournament_select()
        assert isinstance(selected, Individual)
        # It should be one of the existing individuals
        assert selected in self.pop.individuals

    def test_tournament_select_empty_raises(self):
        with pytest.raises(ValueError):
            self.pop.tournament_select()

    def test_get_generation(self):
        for i in range(5):
            self.pop.add(Individual(code=f"g0_{i}", generation=0))
        for i in range(3):
            self.pop.add(Individual(code=f"g1_{i}", generation=1))
        assert len(self.pop.get_generation(0)) == 5
        assert len(self.pop.get_generation(1)) == 3
        assert len(self.pop.get_generation(2)) == 0

    def test_get_stats(self):
        self._add_individuals(4, fitness_fn=lambda i: [0.0, 0.5, 0.5, 1.0][i])
        stats = self.pop.get_stats()
        assert stats["size"] == 4
        assert stats["best_fitness"] == 1.0
        assert abs(stats["avg_fitness"] - 0.5) < 1e-6
        assert stats["diversity"] > 0

    def test_get_stats_empty(self):
        stats = self.pop.get_stats()
        assert stats["size"] == 0
        assert stats["avg_fitness"] == 0.0

    def test_elite_preserves_order(self):
        self._add_individuals(10)
        elites = self.pop.get_elite()
        # Elites should be sorted descending by fitness
        for i in range(len(elites) - 1):
            assert elites[i].fitness >= elites[i + 1].fitness
