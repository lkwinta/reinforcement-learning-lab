import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np

from itertools import accumulate
from bandits import *

class BanditProblem:
    def __init__(self, time_steps: int, bandit: KArmedBandit, learner: BanditLearner):
        self.time_steps: int = time_steps
        self.bandit: KArmedBandit = bandit
        self.learner: BanditLearner = learner
        self.learner.reset(self.bandit.arms(), self.time_steps)

    def run(self) -> list[float]:
        rewards = []
        for _ in range(self.time_steps):
            arm = self.learner.pick_arm()
            reward = self.bandit.reward(arm)
            self.learner.acknowledge_reward(arm, reward)
            rewards.append(reward)
        return rewards


POTENTIAL_HITS = {
    "In Praise of Dreams": 0.8,
    "We Built This City": 0.9,
    "April Showers": 0.5,
    "Twenty Four Hours": 0.3,
    "Dirge for November": 0.1,
}

TIME_STEPS = 1000
TRIALS_PER_LEARNER = 50


def evaluate_learner(learner: BanditLearner) -> None:
    runs_results = []
    for _ in range(TRIALS_PER_LEARNER):
        bandit = TopHitBandit(POTENTIAL_HITS)
        problem = BanditProblem(time_steps=TIME_STEPS, bandit=bandit, learner=learner)
        rewards = problem.run()
        accumulated_rewards = list(accumulate(rewards))
        runs_results.append(accumulated_rewards)

    runs_results = np.array(runs_results)
    mean_accumulated_rewards = np.mean(runs_results, axis=0)
    std_accumulated_rewards = np.std(runs_results, axis=0)
    plt.plot(mean_accumulated_rewards, label=learner.name, color=learner.color)
    plt.fill_between(
        range(len(mean_accumulated_rewards)),
        mean_accumulated_rewards - std_accumulated_rewards,
        mean_accumulated_rewards + std_accumulated_rewards,
        color=learner.color,
        alpha=0.2,
    )


def main():
    learners = [
        RandomLearner(),
        ExploreThenCommitLearner(m=5, color="green"),
        EGreedyLearner(epsilon=0.1, color="red"),
        EGreedyLearner(epsilon=0.0, color="orange"), # full greedy
        EGreedyLearner(epsilon=0.1, bias={arm: 0.1 for arm in POTENTIAL_HITS.keys()}, color="purple"), # more exploration
        UCB1Learner(c=1.0, color="blue"),
        UCB1Learner(c=2.0, color="cyan"),
        GradientLearner(lr=0.3, color="yellow"),
        ThompsonLearner(color="pink"),
    ]
    for learner in learners:
        evaluate_learner(learner)

    plt.xlabel('Czas')
    plt.ylabel('Suma uzyskanych nagród')
    plt.xlim(0, TIME_STEPS)
    plt.ylim(0, TIME_STEPS)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
