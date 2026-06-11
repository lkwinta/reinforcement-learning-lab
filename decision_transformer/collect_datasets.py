from argparse import ArgumentParser

import gymnasium as gym
import torch
from stable_baselines3 import DQN
from tqdm.rich import tqdm

from minari import DataCollector


def collect_dataset(
    dataset_name: str, env_name: str, expert_path: str, n_timesteps: int, seed: int
):
    torch.manual_seed(seed)

    env = DataCollector(gym.make(env_name))
    agent = DQN.load(expert_path, env=env)

    done = True

    for i in tqdm(range(n_timesteps), desc=f"Collecting dataset {dataset_name}"):
        if done:
            obs, _ = env.reset(seed=seed + i)

        action, _ = agent.predict(obs)
        obs, rew, ter, tru, info = env.step(action)

        done = ter or tru

    _dataset = env.create_dataset(
        dataset_id=dataset_name,
        algorithm_name="ExpertPolicy",
        eval_env=env_name,
        author="Łukasz Kwinta",
        author_email="lukaszmind@gmail.com",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env-name", type=str, default="LunarLander-v3")
    parser.add_argument("--n-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1337)

    args = parser.parse_args()

    collect_dataset(
        env_name=args.env_name,
        expert_path="logs/dqn/LunarLander-v3_1/best_model.zip",
        n_timesteps=args.n_timesteps,
        seed=args.seed,
        dataset_name=f"{args.env_name}-full-expert-v0",
    )

    collect_dataset(
        env_name=args.env_name,
        expert_path="logs/dqn/LunarLander-v3_1/rl_model_70000_steps.zip",
        n_timesteps=args.n_timesteps,
        seed=args.seed,
        dataset_name=f"{args.env_name}-medium-expert-v0",
    )
