import os

import minari
import torch
import argparse


from nanodt.agent import NanoDTAgent
from nanodt.utils import seed_libraries


def get_device():
    if torch.backends.mps.is_available():
        return "mps"  # macOS GPU
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def train_dt(dataset_name: str):
    seed = 1234
    seed_libraries(seed)

    save_path = f"output/dt/minari-{dataset_name}-dt.pth"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    minari_dataset = minari.load_dataset(dataset_name)

    device = get_device()
    print(f"Using device: {device}")

    print(minari_dataset.action_space)

    dt_agent = NanoDTAgent(
        device=device,
        K=60,
        max_ep_len=1000,
    )
    dt_agent.learn(
        minari_dataset, learning_rate=1e-4, weight_decay=0.0485, reward_scale=13.194
    )
    dt_agent.save(save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    args = parser.parse_args()

    train_dt(args.dataset_name)
