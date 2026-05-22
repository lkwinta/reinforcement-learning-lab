import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm.rich import tqdm


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, act_dim)

    def forward(self, obs):
        x = obs

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.argmax(self(observation), axis=-1)


def evaluate():
    policy_net = PolicyNetwork(obs_dim=8, act_dim=4)
    policy_net.load_state_dict(torch.load("output/bc/policy_net.pth"))
    policy_net.eval()

    env = gym.make("LunarLander-v3")

    rewards = []

    for _ in tqdm(range(1000)):
        obs, _ = env.reset()
        done = False
        accumulated_rew = 0
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = policy_net.act(obs_tensor)
            obs, rew, ter, tru, _ = env.step(action.item())
            done = ter or tru
            accumulated_rew += rew

        rewards.append(accumulated_rew)
        print("Accumulated rew: ", accumulated_rew)

    print("Average reward over 1000 episodes: ", np.mean(rewards))

    env.close()


if __name__ == "__main__":
    evaluate()
