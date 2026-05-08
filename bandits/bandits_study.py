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

def evaluate_learner_regret(learner: BanditLearner) -> None:
    all_rewards = []
    all_optimal_rewards = []

    for _ in range(TRIALS_PER_LEARNER):
        potential_hits = random_potential_hits(len(POTENTIAL_HITS))

        bandit = TopHitBandit(potential_hits)
        problem = BanditProblem(time_steps=TIME_STEPS, bandit=bandit, learner=learner)
        rewards = problem.run()

        optimal_expected_reward = max(potential_hits.values())
        optimal_rewards = np.full(TIME_STEPS, optimal_expected_reward)

        all_rewards.append(rewards)
        all_optimal_rewards.append(optimal_rewards)
    
    all_rewards = np.array(all_rewards)
    all_optimal_rewards = np.array(all_optimal_rewards)

    mean_regret = regret_calculation(all_rewards, all_optimal_rewards)
    plt.plot(mean_regret, label=learner.name, color=learner.color)

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
    plt.plot(mean_accumulated_rewards, label=learner.name, color=learner.color)

def save_plot(path: str, title: str, y_lim: bool = True, y_label: str = 'Suma uzyskanych nagród') -> None:
    plt.xlabel('Czas')
    plt.ylabel(y_label)
    plt.xlim(0, TIME_STEPS)
    if y_lim:
        plt.ylim(0, TIME_STEPS)
    plt.legend()
    plt.title(title)
    plt.savefig(path)

explore_then_commit_grid = {
    "m": [1, 5, 10, 20],
}
e_greedy_grid = {
    "epsilon": [0.0, 0.01, 0.1, 0.5],
}
ucb1_grid = {
    "c": [0.0, 0.5, 1.0, 2.0],
}
gradient_grid = {
    "lr": [0.1, 0.2, 0.3, 0.4, 0.5, 0.8],
}

colors = ["green", "red", "blue", "yellow", "cyan", "magenta", "orange", "purple", "pink"]

def parametric_study():
    learners = [
        (ExploreThenCommitLearner, explore_then_commit_grid),
        (EGreedyLearner, e_greedy_grid),
        (UCB1Learner, ucb1_grid),
        (GradientLearner, gradient_grid),
    ]

    for learner_class, grid in learners:
        plt.clf()
        idx = 0
        for param in grid.items():
            key, value_list = param
            for value in value_list:
                learner = learner_class(**{key: value, "color": colors[idx]})
                evaluate_learner(learner)
                idx += 1

        save_plot(f"plots/{learner_class.__name__.lower()}.png", f"{learner_class.__name__}", y_lim=True)

        plt.clf()
        idx = 0
        for param in grid.items():
            key, value_list = param
            for value in value_list:
                learner = learner_class(**{key: value, "color": colors[idx]})
                evaluate_learner_regret(learner)
                idx += 1

        save_plot(f"plots/{learner_class.__name__.lower()}_regret.png", f"{learner_class.__name__} - Regret", y_lim=False, y_label='Regret')

def main():
    parametric_study()

if __name__ == '__main__':
    main()
