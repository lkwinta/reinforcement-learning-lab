import argparse

import numpy as np
from nanodt.agent import NanoDTAgent
import gymnasium as gym
import torch
from tqdm.rich import tqdm


def evaluate(dt_path: str):
    env = gym.make("LunarLander-v3")
    agent = NanoDTAgent.load(
        dt_path, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    rewards = []

    for _ in tqdm(range(1000)):
        agent.reset(target_return=160)
        obs, info = env.reset()
        done = False
        accumulated_rew = 0
        prev_rew = None

        while not done:
            action = agent.act(obs, rew=prev_rew)
            obs, rew, ter, tru, info = env.step(action)
            done = ter or tru
            accumulated_rew += rew

            prev_rew = torch.tensor(rew)  # Initialize prev_rew with the first reward

        rewards.append(accumulated_rew)
        # print("Accumulated rew: ", accumulated_rew)
        # print("Success: ", info["is_success"])

    print(f"{dt_path}: Average reward over 1000 episodes: {np.mean(rewards)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt-path", type=str, required=True)
    args = parser.parse_args()

    evaluate(args.dt_path)
