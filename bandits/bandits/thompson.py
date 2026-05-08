from . import BanditLearner
import random

class ThompsonLearner(BanditLearner):
    def __init__(self, color: str = "pink"):
        self.name = "Thompson"
        self.color = color
        self.arms: list[str] = []
        self.alpha = {}
        self.beta = {}

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.alpha = {arm: 1.0 for arm in arms}
        self.beta = {arm: 1.0 for arm in arms}

    def pick_arm(self) -> str:
        samples = {
            arm: random.betavariate(self.alpha[arm], self.beta[arm])
            for arm in self.arms
        }
        return max(self.arms, key=lambda arm: samples[arm])

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
