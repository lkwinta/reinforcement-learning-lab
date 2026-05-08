import matplotlib.pyplot as plt
import numpy as np

from itertools import accumulate
from bandits import *

POTENTIAL_HITS = {
    "In Praise of Dreams": 0.8,
    "We Built This City": 0.9,
    "April Showers": 0.5,
    "Twenty Four Hours": 0.3,
    "Dirge for November": 0.1,
}

def update_probabilites(potential_hits: dict[str, float]) -> dict[str, float]:
    updated_hits = {
        song: max(0.0, min(1.0, prob + np.random.normal(0, 0.05)))
        for song, prob in potential_hits.items()
    }
    return updated_hits

class BanditProblem:
    def __init__(self, time_steps: int, bandit: KArmedBandit, learner: BanditLearner, change_probs: bool = False):
        self.time_steps: int = time_steps
        self.bandit: KArmedBandit = bandit
        self.learner: BanditLearner = learner
        self.change_probs: bool = change_probs
        self.learner.reset(self.bandit.arms(), self.time_steps)

    def run(self) -> list[float]:
        rewards = []
        optimal_rewards = []

        for _ in range(self.time_steps):
            if self.change_probs:
                self.bandit.potential_hits = update_probabilites(self.bandit.potential_hits)

            optimal_expected_reward = max(self.bandit.potential_hits.values())
            
            arm = self.learner.pick_arm()
            reward = self.bandit.reward(arm)

            self.learner.acknowledge_reward(arm, reward)

            rewards.append(reward)
            optimal_rewards.append(optimal_expected_reward)

        return rewards, optimal_rewards

TIME_STEPS = 1000
TRIALS_PER_LEARNER = 50

def random_potential_hits(k: int = 5) -> dict[str, float]:
    probabilities = np.random.random(k)
    return {
        f"Song {i}": p
        for i, p in enumerate(probabilities)
    }

def regret_calculation(rewards_matrix, optimal_rewards_matrix):
    regrets = np.cumsum(optimal_rewards_matrix - rewards_matrix, axis=1)
    mean_regret = np.mean(regrets, axis=0)
    return mean_regret

def evaluate_learner_regret(learner: BanditLearner, change_probs: bool = False) -> None:
    all_rewards = []
    all_optimal_rewards = []

    for _ in range(TRIALS_PER_LEARNER):
        potential_hits = random_potential_hits(len(POTENTIAL_HITS))

        bandit = TopHitBandit(potential_hits)
        problem = BanditProblem(time_steps=TIME_STEPS, bandit=bandit, learner=learner, change_probs=change_probs)
        rewards, optimal_rewards = problem.run()

        all_rewards.append(rewards)
        all_optimal_rewards.append(optimal_rewards)
    
    all_rewards = np.array(all_rewards)
    all_optimal_rewards = np.array(all_optimal_rewards)

    mean_regret = regret_calculation(all_rewards, all_optimal_rewards)
    plt.plot(mean_regret, label=learner.name, color=learner.color)

def evaluate_learner(learner: BanditLearner, change_probs: bool = False) -> None:
    runs_results = []
    for _ in range(TRIALS_PER_LEARNER):
        potential_hits = random_potential_hits(len(POTENTIAL_HITS))

        bandit = TopHitBandit(potential_hits)
        problem = BanditProblem(time_steps=TIME_STEPS, bandit=bandit, learner=learner, change_probs=change_probs)
        rewards, _ = problem.run()
        accumulated_rewards = list(accumulate(rewards))
        runs_results.append(accumulated_rewards)

    runs_results = np.array(runs_results)
    mean_accumulated_rewards = np.mean(runs_results, axis=0)
    plt.plot(mean_accumulated_rewards, label=learner.name, color=learner.color)

def study(change_probs: bool = False) -> None:
    learners = [
        RandomLearner(),
        ExploreThenCommitLearner(m=10, color="green"),
        EGreedyLearner(epsilon=0.1, color="red"),
        UCB1Learner(c=0.5, color="blue"),
        GradientLearner(lr=0.8, color="yellow"),
        ThompsonLearner(color="pink"),
    ]

    for learner in learners:
        evaluate_learner(learner, change_probs=change_probs)

    plt.xlabel('Czas')
    plt.ylabel('Suma uzyskanych nagród')
    plt.xlim(0, TIME_STEPS)
    plt.ylim(0, TIME_STEPS)
    plt.legend()
    plt.savefig(f"bandits{'_drifting' if change_probs else ''}_rewards.png")
    plt.clf()

    for learner in learners:
        evaluate_learner_regret(learner, change_probs=change_probs)

    plt.xlabel('Czas')
    plt.ylabel('Średni skumulowany regret')
    plt.xlim(0, TIME_STEPS)
    plt.legend()
    plt.savefig(f"bandits{'_drifting' if change_probs else ''}_regret.png")
    plt.clf()

def main():
    study(change_probs=False)
    study(change_probs=True)

if __name__ == '__main__':
    main()
