import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from tqdm.rich import tqdm


def evaluate():
    env = gym.make("LunarLander-v3")
    agent = DQN.load("logs/dqn/LunarLander-v3_1/LunarLander-v3.zip")

    rewards = []

    for _ in tqdm(range(1000)):
        obs, info = env.reset()
        done = False
        accumulated_rew = 0
        while not done:
            action, _ = agent.predict(obs)
            obs, rew, ter, tru, info = env.step(action)
            done = ter or tru
            accumulated_rew += rew

        rewards.append(accumulated_rew)
        print("Accumulated rew: ", accumulated_rew)
        # print("Success: ", info["is_success"])

    print("Average reward over 1000 episodes: ", np.mean(rewards))


if __name__ == "__main__":
    evaluate()
