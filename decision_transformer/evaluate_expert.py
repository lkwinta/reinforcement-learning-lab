import argparse

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from tqdm.rich import tqdm


def evaluate(expert_path: str):
    env = gym.make("LunarLander-v3")
    agent = DQN.load(expert_path)

    rewards = []

    for _ in tqdm(range(1000)):
        obs, info = env.reset()
        done = False
        accumulated_rew = 0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, rew, ter, tru, info = env.step(action)
            done = ter or tru
            accumulated_rew += rew

        rewards.append(accumulated_rew)
        # print("Accumulated rew: ", accumulated_rew)
        # print("Success: ", info["is_success"])

    print(f"{expert_path}: Average reward over 1000 episodes: {np.mean(rewards)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-path", type=str, required=True)
    args = parser.parse_args()

    evaluate(args.expert_path)
