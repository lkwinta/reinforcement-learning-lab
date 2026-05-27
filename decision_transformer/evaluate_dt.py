import numpy as np
from nanodt.agent import NanoDTAgent
import gymnasium as gym
import torch
from tqdm.rich import tqdm


def evaluate():
    env = gym.make("LunarLander-v3")
    agent = NanoDTAgent.load("output/dt/minari-LunarLander-v3-expert-v0.pth")

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
        print("Accumulated rew: ", accumulated_rew)
        # print("Success: ", info["is_success"])

    print("Average reward over 1000 episodes: ", np.mean(rewards))


if __name__ == "__main__":
    evaluate()
