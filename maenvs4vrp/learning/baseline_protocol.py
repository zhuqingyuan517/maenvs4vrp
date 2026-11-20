"""Shared settings for the large-scale DTOPTW baseline experiments.

The paper compares methods on two scales and requires the environment
constraints, decision horizon, and local action-space size to remain aligned.
Keeping the values here avoids silent differences between training and
baseline-evaluation command lines.
"""

from __future__ import annotations

from dataclasses import dataclass


PROPOSED_K = 30
PROPOSED_INDUCING_POINTS = 32


def max_steps_for_scale(num_nodes: int) -> int:
    """Return the paper's maximum decision-step budget for a node scale."""
    return 1200 if int(num_nodes) >= 1000 else 600


def resolve_max_steps(num_nodes: int, max_steps: int | None) -> int:
    """Resolve ``0``/``None`` to the scale-specific paper default."""
    if max_steps is None or int(max_steps) <= 0:
        return max_steps_for_scale(num_nodes)
    return int(max_steps)


@dataclass(frozen=True)
class ExperimentProtocol:
    """Controls that must match across learned and heuristic baselines."""

    horizon: int = 480
    max_flight_dist: float = 300.0
    capacity: int = 200
    dynamic_ratio: float = 0.5
    time_window_ratio: float = 0.5
    local_k: int = PROPOSED_K
    inducing_points: int = PROPOSED_INDUCING_POINTS
    max_steps_500: int = 600
    max_steps_1000: int = 1200

    def max_steps(self, num_nodes: int) -> int:
        return self.max_steps_1000 if int(num_nodes) >= 1000 else self.max_steps_500


DEFAULT_PROTOCOL = ExperimentProtocol()
