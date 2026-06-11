import os
import argparse

import minari
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.utils.data import DataLoader
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


def collate_fn(batch):
    return {
        "id": torch.Tensor([x.id for x in batch]),
        # "seed": torch.Tensor([x.seed for x in batch]),
        # "total_steps": torch.Tensor([x.total_steps for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.observations, dtype=torch.float32) for x in batch],
            batch_first=True,
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions, dtype=torch.long) for x in batch],
            batch_first=True,
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch], batch_first=True
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations) for x in batch], batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch], batch_first=True
        ),
    }


def train_bc(dataset_name: str):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    minari_dataset = minari.load_dataset(dataset_name)
    dataloader = DataLoader(
        minari_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn
    )

    env = minari_dataset.recover_environment()
    observation_space = env.observation_space
    action_space = env.action_space

    assert isinstance(observation_space, spaces.Box)
    assert isinstance(action_space, spaces.Discrete)

    obs_dim = observation_space.shape[0]
    act_dim = action_space.n

    print(observation_space.shape, action_space.n)

    policy_net = PolicyNetwork(obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters())
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100).to(device)

    num_epochs = 100

    for epoch in tqdm(range(num_epochs)):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}"):
            logits = policy_net(batch["observations"][:, :-1].to(device))
            a_hat = batch["actions"].to(device)

            B, T, A = logits.shape

            loss = loss_fn(
                logits.reshape(B * T, A),
                a_hat.reshape(B * T),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}")

    # Save the model
    policy_net.eval()
    policy_net.cpu()

    os.makedirs("output/bc", exist_ok=True)
    torch.save(policy_net.state_dict(), f"output/bc/{dataset_name}_policy_net.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    args = parser.parse_args()

    train_bc(args.dataset_name)
