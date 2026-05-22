from . import BanditLearner

import random


class EGreedyLearner(BanditLearner):
    def __init__(
        self, epsilon: float = 0.1, bias: dict[str, float] = None, color: str = "red"
    ):
        self.name = f"EGreedy (epsilon={epsilon})"
        self.color = color
        self.epsilon = epsilon
        self.arms: list[str] = []
        self.bias = bias
        self.Q = {}
        self.counts = {}

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.Q = {arm: 0 for arm in self.arms}
        if self.bias is not None:
            for arm in self.arms:
                self.Q[arm] = self.bias.get(arm, 0)
        self.counts = {arm: 0 for arm in self.arms}

    def pick_arm(self) -> str:
        arm = None

        if random.random() < self.epsilon:
            arm = random.choice(self.arms)
        else:
            best_value = max(self.Q[arm] for arm in self.arms)
            best_arms = [arm for arm in self.arms if self.Q[arm] == best_value]
            arm = random.choice(best_arms)

        self.counts[arm] += 1
        return arm

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.Q[arm] += (reward - self.Q[arm]) / self.counts[arm]
