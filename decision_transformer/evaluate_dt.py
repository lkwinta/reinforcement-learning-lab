import numpy as np
from nanodt.agent import NanoDTAgent
import gymnasium as gym


def evaluate():
    env = gym.make("LunarLander-v3")
    agent = NanoDTAgent.load("output/dt/minari-LunarLander-v3-expert-v0.pth")

    rewards = []

    for _ in range(20):
        agent.reset(target_return=200)
        obs, info = env.reset()
        done = False
        accumulated_rew = 0
        while not done:
            action = agent.act(obs)
            obs, rew, ter, tru, info = env.step(action)
            done = ter or tru
            accumulated_rew += rew

        rewards.append(accumulated_rew)
        print("Accumulated rew: ", accumulated_rew)
        # print("Success: ", info["is_success"])

    print("Average reward over 20 episodes: ", np.mean(rewards))


if __name__ == "__main__":
    evaluate()
