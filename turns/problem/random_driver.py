from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np

from .problem import (
    Action,
    Driver,
    State,
    POTENTIAL_ACTIONS_ARRAY,
    action_from_numba,
    available_actions_jit,
    state_to_numba,
)


class RandomDriver(Driver):
    def __init__(self, max_learning_steps: int = 500) -> None:
        self.current_step: int = 0
        self.max_learning_steps: int = max_learning_steps

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        return self._random_action(state)

    def control(self, state: State, last_reward: int) -> Action:
        self.current_step += 1

        if last_reward == 0:
            return Action(0, 0)

        return self._random_action(state)

    def _random_action(self, state: State) -> Action:
        actions = available_actions_jit(
            state_to_numba(state),
            POTENTIAL_ACTIONS_ARRAY,
        )

        index = np.random.randint(actions.shape[0])
        return action_from_numba((actions[index, 0], actions[index, 1]))

    def finished_learning(self) -> bool:
        return self.current_step > self.max_learning_steps
