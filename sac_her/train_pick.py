# A script to train a Soft Actor-Critic (SAC) agent with Hindsight Experience Replay (HER) on a specified gym environment.
# By default it uses the PandaReach-v3 environment from the panda_gym package.
# But it can be easily modified to use any other gym environment, including those from the gymnasium_robotics package.
# Just uncomment the import statement for gymnasium_robotics and register the environments and use for example the FetchReach-v3 environment.

# Also give it a try on push and pick and place tasks: PandaPush-v3, PandaPickAndPlace-v3, FetchPush-v3, FetchPickAndPlace-v3

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
# Uncomment the following line to use gymnasium_robotics environments
# import gymnasium_robotics
import panda_gym
import torch

# Uncomment the following lines to register gymnasium_robotics environments
# gym.register_envs(gymnasium_robotics)

from asdf.algos import SAC
from asdf.buffers import HerReplayBuffer, DictReplayBuffer
from asdf.extractors import DictExtractor
from asdf.loggers import JSONLogger
from asdf.policies import MlpPolicy

# There are two challenges in this exercise:
# 1. Implement the Hindsight Experience Replay (HER) algorithm.
#    This is done in the HerReplayBuffer class.
# 2. Improve the SAC algorithm with an automatically adjusted temperature (alpha) parameter.
#    This is done in the SAC class.
def main(config: dict) -> None:

    env_id = config["env"]
    load = config["load"]
    render = config["render"]

    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU")
    else:
        print("Using CPU")
        device = "cpu"

    env = gym.make(env_id)

    policy = MlpPolicy(
        env.observation_space,
        env.action_space,
        hidden_sizes=[256, 256, 256, 256],
        extractor_type=DictExtractor,
    )
    policy.to(device)


    buffer_type = config.get("buffer_type", "her")
    if buffer_type == "her":
        her_strategy = config.get("her_strategy", "future")
        n_sampled_goal = config.get("her_n_sampled_goal", 3)

        print(f"Using HER replay buffer with strategy '{her_strategy}' and {n_sampled_goal} sampled goals per transition")

        buffer = HerReplayBuffer(
            env=env,
            size=1_000_000,
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=her_strategy,
            device=device,
        )
    elif buffer_type == "dict":
        buffer = DictReplayBuffer(
            env=env,
            size=1_000_000,
            device=device,
        )
    
    logger = JSONLogger(
        save_path=Path(f"{config['logs_dir']}/sac_her_log.json"),
        save_every=20000
    )

    alpha = config["alpha"]

    algo = SAC(
        env,
        policy=policy,
        buffer=buffer,
        update_every=1,
        update_after=40_000,
        batch_size=128,
        # alpha="auto", # use automatic alpha adjustment (uncoment when implemented)
        alpha=alpha, # use fixed alpha (comment out when implementing automatic alpha adjustment)
        gamma=0.98,
        # polyak=0.95,
        lr=1e-4,
        logger=logger,
        max_episode_len=env.spec.max_episode_steps,
        start_steps=40_000,
    )

    if load is not None:
        algo.load(load)

    if not render:
        algo.train(n_steps=config["n_steps"], log_interval=2000)
        env.close()
        logger.save()
        algo.save(f"{config['logs_dir']}/sac_her_final.pth")
        print("Training completed and model saved as sac_her_final.pth")

        env = gym.make(env_id)
        eval_results = algo.test(env, n_episodes=10, episode_sleep=0, render_sleep=0)
        json.dump(eval_results, open(f"{config['logs_dir']}/sac_her_final_eval.json", "w"), indent=4)
        env.close()
    else:
        policy.cpu()
        env = gym.make(env_id, render_mode="human")
        results = algo.test(env, n_episodes=10, episode_sleep=1, render_sleep=1 / 20)
        env.close()
        print(f"Test reward {results['mean_ep_ret']}, Test episode length: {results['mean_ep_len']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load", type=str, default=None, help="Path to a saved model to load"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment without training"
    )
    parser.add_argument(
        "--logs_dir", type=str, default="logs", help="Directory to save training logs"
    )
    parser.add_argument(
        "--alpha", default=0.95, help="Temperature parameter for SAC (use 'auto' for automatic adjustment)"
    )
    parser.add_argument(
        "--buffer_type", choices=["her", "dict"], default="her", help="Type of replay buffer to use"
    )
    parser.add_argument(
        "--her_strategy", choices=["future", "final", "episode"], default="future", help="HER goal selection strategy"
    )
    parser.add_argument(
        "--her_n_sampled_goal", type=int, default=3, help="Number of HER goals to sample per transition"
    )
    parser.add_argument(
        "--n_steps", type=int, default=100_000, help="Number of training steps"
    )

    args = parser.parse_args()

    config = {
        "env": "PandaPick-v3",
        "load": args.load,
        "render": args.render,
        "logs_dir": args.logs_dir,
        "n_steps": args.n_steps,
        "alpha": args.alpha,
        "buffer_type": args.buffer_type,
        "her_strategy": args.her_strategy,
        "her_n_sampled_goal": args.her_n_sampled_goal,
    }

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)

    json.dump(config, open(f"{args.logs_dir}/config.json", "w"), indent=4)

    main(config)
