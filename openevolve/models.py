"""Data models for OpenEvolve."""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Individual:
    """A single candidate solution in the population."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    code: str = ""
    fitness: float = 0.0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=lambda: {"created_at": time.time()})


@dataclass
class SandboxResult:
    """Result of executing code in the sandbox."""

    success: bool
    output: str = ""
    error: str = ""
    execution_time: float = 0.0


@dataclass
class EvolutionConfig:
    """Configuration for the evolutionary process."""

    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.3
    elite_ratio: float = 0.2
    tournament_size: int = 3
    timeout_seconds: int = 10
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = ""
    llm_api_key: str = ""


@dataclass
class EvolutionResult:
    """Result of running an evolutionary optimization."""

    best_individual: Individual
    generation_stats: list[dict[str, Any]] = field(default_factory=list)
    total_evaluations: int = 0
    elapsed_seconds: float = 0.0
