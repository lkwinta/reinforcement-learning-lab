from .bandits import BanditLearner
import numpy as np


class GradientLearner(BanditLearner):
    def __init__(self, lr: float = 0.1, color: str = "yellow"):
        self.name = f"Gradient (lr={lr})"
        self.color = color
        self.arms: list[str] = []
        self.H = np.array([])
        self.lr = lr
        self.average_reward = 0.0
        self.reward_count = 0

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.H = np.zeros(len(arms))
        self.average_reward = 0.0
        self.reward_count = 0

    def pick_arm(self) -> str:
        probabilities = self._softmax(self.H)
        arm = np.random.choice(self.arms, p=probabilities)
        return arm

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        arm_index = self.arms.index(arm)
        probs = self._softmax(self.H)

        ones = np.zeros(len(self.arms))
        ones[arm_index] = 1

        self.H += self.lr * (reward - self.average_reward) * (ones - probs)

        self.reward_count += 1
        self.average_reward += (reward - self.average_reward) / self.reward_count

    def _softmax(self, values: np.array) -> np.array:
        exp_values = np.exp(values - np.max(values))  # stabilnosc numeryczna
        return exp_values / np.sum(exp_values)
