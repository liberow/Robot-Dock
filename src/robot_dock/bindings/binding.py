from dataclasses import dataclass
from typing import Callable

from robot_dock.protocol import ActionChunk, Observation


def _observation_identity(value: Observation) -> Observation:
    return value


def _action_identity(value: ActionChunk) -> ActionChunk:
    return value


@dataclass(frozen=True)
class PolicyBinding:
    """Explicit, ordinary functions; no transform DSL or automatic shape guessing."""

    name: str = "identity"
    prepare_observation: Callable[[Observation], Observation] = _observation_identity
    prepare_actions: Callable[[ActionChunk], ActionChunk] = _action_identity
