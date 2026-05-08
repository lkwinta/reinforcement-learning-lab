from .bandits import BanditLearner

class ExploreThenCommitLearner(BanditLearner):
    def __init__(self, m: int = 5, color: str = "blue"):
        self.name = f"ExploreThenCommit (m={m})"
        self.color = color
        self.arms: list[str] = []
        self.rewards = {}
        self.round = 0
        self.m = m
        self.max_reward_arm = None

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.rewards = { arm: 0 for arm in self.arms }
        self.round = 0

    def pick_arm(self) -> str:
        arm_index = None
        if self.round < len(self.arms) * self.m:
            arm_index = self.round % len(self.arms)
        else:
            if self.round == len(self.arms) * self.m:
                self.max_reward_arm = self.arms.index(max(self.rewards, key=self.rewards.get))
            arm_index = self.max_reward_arm

        self.round += 1
        
        return self.arms[arm_index]
    
    def acknowledge_reward(self, arm: str, reward: float) -> None:
        # nie musimy liczyć średniej, bo każde ramię jest testowane dokładnie m razy, 
        # więc wystarczy zliczać sumę nagród
        self.rewards[arm] += reward
