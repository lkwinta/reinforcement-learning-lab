from __future__ import annotations

import random

from . import Action, Driver, State, available_actions


class RandomDriver(Driver):
    def __init__(self, max_learning_steps: int = 500) -> None:
        self.current_step: int = 0
        self.max_learning_steps: int = max_learning_steps

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        return random.choice(available_actions(state))

    def control(self, state: State, last_reward: int) -> Action:
        self.current_step += 1
        return random.choice(available_actions(state))

    def finished_learning(self) -> bool:
        return self.current_step > self.max_learning_steps
