from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .utils import combined_shape


class BaseBuffer(ABC):
    @abstractmethod
    def __init__(
        self, env: gym.Env, size: int = 100000, device: Optional[torch.device] = None
    ) -> None:
        self.device = device

        self.actions = torch.zeros(
            combined_shape(size, env.action_space.shape),
            dtype=torch.float32,
            device=device,
        )
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.terminations = torch.zeros(size, dtype=torch.float32, device=device)
        self.truncations = torch.zeros(size, dtype=torch.float32, device=device)
        self.infos = np.empty((size,), dtype=object)
        self._ptr, self.size, self.max_size = 0, 0, size

    def store(
        self,
        observation: Union[NDArray, dict[str, NDArray]],
        action: NDArray,
        reward: float,
        next_observation: Union[NDArray, dict[str, NDArray]],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self._store_observations(observation, next_observation)
        self.actions[self._ptr] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self._ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.terminations[self._ptr] = torch.as_tensor(terminated, dtype=torch.float32)
        self.truncations[self._ptr] = torch.as_tensor(truncated, dtype=torch.float32)
        self.infos[self._ptr] = info
        self._ptr = (self._ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    @abstractmethod
    def _store_observations(
        self,
        observation: Union[NDArray, dict[str, NDArray]],
        next_observation: Union[NDArray, dict[str, NDArray]],
    ) -> None: ...

    def sample_batch(
        self, batch_size: int = 32
    ) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        idxs = torch.randint(0, self.size, size=(batch_size,))
        # idxs = np.random.randint(0, self.size, size=batch_size)
        return self.batch(idxs)

    def batch(self, idxs: Tensor) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        data = dict(
            action=self.actions[idxs],
            reward=self.rewards[idxs],
            terminated=self.terminations[idxs],
            truncated=self.truncations[idxs],
            info=self.infos[idxs],
        )
        observations = self._observations_batch(idxs)
        data.update(observations)

        return data

    @abstractmethod
    def _observations_batch(
        self, idxs: Tensor
    ) -> dict[str, Union[Tensor, dict[str, Tensor]]]: ...

    def start_episode(self):
        pass

    def end_episode(self):
        pass

    def clear(self):
        self.actions.zero_()
        self.rewards.zero_()
        self.terminations.zero_()
        self.truncations.zero_()
        self.infos.fill(None)
        self._ptr, self.size = 0, 0


class DictReplayBuffer(BaseBuffer):
    """
    A dictionary experience replay buffer for off-policy agents.
    """

    def __init__(
        self, env: gym.Env, size: int = 100000, device: Optional[torch.device] = None
    ):
        assert isinstance(env.observation_space, gym.spaces.Dict)
        super().__init__(env=env, size=size, device=device)

        obs_space = {
            k: combined_shape(size, v.shape) for k, v in env.observation_space.items()
        }

        self.observations: dict[str, Tensor] = {
            k: torch.zeros(obs_space[k], dtype=torch.float32, device=device)
            for k, v in env.observation_space.items()
        }
        self.next_observations: dict[str, Tensor] = {
            k: torch.zeros(obs_space[k], dtype=torch.float32, device=device)
            for k, v in env.observation_space.items()
        }

    def _store_observations(
        self,
        observation: dict[str, NDArray],
        next_observation: dict[str, NDArray],
    ) -> None:
        for k in observation.keys():
            self.observations[k][self._ptr] = torch.as_tensor(
                observation[k], dtype=torch.float32
            )
        for k in next_observation.keys():
            self.next_observations[k][self._ptr] = torch.as_tensor(
                next_observation[k], dtype=torch.float32
            )

    def _observations_batch(self, idxs: Tensor) -> dict[str, dict[str, Tensor]]:
        return dict(
            observation={k: v[idxs] for k, v in self.observations.items()},
            next_observation={k: v[idxs] for k, v in self.next_observations.items()},
        )




class HerReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        env: gym.Env,
        size: int = 100000,
        device: Optional[torch.device] = None,
        n_sampled_goal: int = 1,
        goal_selection_strategy: str = "final",
    ):
        super().__init__(env=env, size=size, device=device)
        self.env = env
        self.n_sampled_goal = n_sampled_goal
        self.selection_strategy = goal_selection_strategy

        self.current_episode = []

    def _sample_goal(self, transition_idx: int) -> dict[str, Any]:
        if self.selection_strategy == "final":
            goal = self.current_episode[-1]["next_observation"]["achieved_goal"]
        elif self.selection_strategy == "future":
            future_transitions = self.current_episode[transition_idx + 1:]
            if len(future_transitions) > 0:
                future_transition_idx = np.random.randint(len(future_transitions))
                future_transition = future_transitions[future_transition_idx]
                goal = future_transition["next_observation"]["achieved_goal"]
            else:
                goal = self.current_episode[-1]["next_observation"]["achieved_goal"]
        elif self.selection_strategy == "episode":
            episode_transitions = self.current_episode
            random_transition_idx = np.random.randint(len(episode_transitions))
            random_transition = episode_transitions[random_transition_idx]
            goal = random_transition["next_observation"]["achieved_goal"]
        else:
            raise ValueError(f"Invalid goal selection strategy: {self.selection_strategy}")
        
        return goal
    
    # compute distance from https://github.com/qgallouedec/panda-gym/blob/master/panda_gym/utils.py#L4
    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the distance between two array. This function is vectorized.

        Args:
            a (np.ndarray): First array.
            b (np.ndarray): Second array.

        Returns:
            np.ndarray: The distance between the arrays.
        """
        assert a.shape == b.shape
        dist = np.linalg.norm(a - b, axis=-1)
        # round at 1e-6 (ensure determinism and avoid numerical noise)
        return np.round(dist, 6)
    
    # Compute reward from https://github.com/qgallouedec/panda-gym/blob/master/panda_gym/envs/tasks/reach.py#L59
    def _compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict[str, Any] = {}, sparse=True) -> np.ndarray:
        d = HerReplayBuffer._distance(achieved_goal, desired_goal)
        if sparse:
            # TODO: make the threshold global const
            return -np.array(d > 0.05, dtype=np.float32)
        else:
            return -d.astype(np.float32)
             
    def start_episode(self):
        self.current_episode = []

    def end_episode(self):
        for transition_idx, transition in enumerate(self.current_episode):
            super().store(
                observation=transition["observation"],
                action=transition["action"],
                reward=transition["reward"],
                next_observation=transition["next_observation"],
                terminated=transition["terminated"],
                truncated=transition["truncated"],
                info=transition["info"],
            )

            for _ in range(self.n_sampled_goal):
                goal = self._sample_goal(transition_idx)
                
                new_observation = transition["observation"].copy()
                new_observation["desired_goal"] = goal
                new_next_observation = transition["next_observation"].copy()
                new_next_observation["desired_goal"] = goal

                new_reward = self._compute_reward(
                    achieved_goal=new_next_observation["achieved_goal"],
                    desired_goal=goal,
                    info=transition["info"],
                )
                super().store(
                    observation=new_observation,
                    action=transition["action"],
                    reward=new_reward,
                    next_observation=new_next_observation,
                    terminated=transition["terminated"],
                    truncated=transition["truncated"],
                    info=transition["info"],
                )

        self.current_episode = []

    def store(
        self,
        observation: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_observation: dict[str, torch.Tensor],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ):
        self.current_episode.append({
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        })

        # if terminated or truncated:
        #     self.end_episode()
