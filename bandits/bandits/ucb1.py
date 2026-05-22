from . import BanditLearner
import numpy as np
import random


class UCB1Learner(BanditLearner):
    def __init__(self, c: float = 1.0, color: str = "green"):
        self.name = f"UCB1 (c={c})"
        self.color = color
        self.c = c
        self.arms: list[str] = []
        self.counts = {}
        self.Q = {}
        self.rounds = 0

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.counts = {arm: 0 for arm in self.arms}
        self.Q = {arm: 0 for arm in self.arms}

    def pick_arm(self) -> str:
        weights = {}
        for arm in self.arms:
            if self.counts[arm] == 0:
                weights[arm] = float("inf")
            else:
                weights[arm] = self.Q[arm] + self.c * np.sqrt(
                    np.log(self.rounds) / self.counts[arm]
                )

        self.rounds += 1

        best_weight = max(weights.values())
        best_arms = [arm for arm in self.arms if weights[arm] == best_weight]
        arm = random.choice(best_arms)
        self.counts[arm] += 1
        return arm

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.Q[arm] += (reward - self.Q[arm]) / self.counts[arm]
