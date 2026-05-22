from __future__ import annotations

import numpy as np
import collections
import pickle
from pathlib import Path
import sklearn.preprocessing as skl_preprocessing

from . import Action, Driver, State, available_actions

ALMOST_INFINITE_STEP = 1000000


class OffPolicyNStepSarsaDriver(Driver):
    def __init__(
        self,
        step_size: float,
        step_no: int,
        experiment_rate: float,
        discount_factor: float,
        max_learning_steps: int = 500,
        use_speeding_policy: bool = False,
        speeding_rate: float = 0.2,
        use_importance_sampling: bool = True,
    ) -> None:
        self.step_size: float = step_size
        self.step_no: int = step_no
        self.experiment_rate: float = experiment_rate
        self.discount_factor: float = discount_factor
        self.max_learning_steps: int = max_learning_steps
        self.q: dict[tuple[State, Action], float] = collections.defaultdict(float)
        self.current_step: int = 0
        self.final_step: int = ALMOST_INFINITE_STEP
        self.finished: bool = False
        self.states: dict[int, State] = dict()
        self.actions: dict[int, Action] = dict()
        self.rewards: dict[int, int] = dict()
        self.use_speeding_policy = use_speeding_policy
        self.speeding_rate = speeding_rate
        self.use_importance_sampling = use_importance_sampling

        self.backup_step_size = step_size
        self.backup_experiment_rate = experiment_rate
        self.backup_speeding_rate = speeding_rate
        self.backup_use_speeding_policy = use_speeding_policy

        if self.experiment_rate + self.speeding_rate > 1.0:
            raise ValueError("experiment_rate + speeding_rate must be <= 1.0")

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        self.states[self._access_index(self.current_step)] = state

        policy = (
            self.speeding_policy
            if self.use_speeding_policy
            else self.epsilon_greedy_policy
        )

        action = self._select_action(policy(state, available_actions(state)))
        self.actions[self._access_index(self.current_step)] = action
        self.final_step = ALMOST_INFINITE_STEP
        self.finished = False
        return action

    def control(self, state: State, last_reward: int) -> Action:
        if self.current_step < self.final_step:
            self.rewards[self._access_index(self.current_step + 1)] = last_reward
            self.states[self._access_index(self.current_step + 1)] = state
            if self.final_step == ALMOST_INFINITE_STEP and (
                last_reward == 0 or self.current_step == self.max_learning_steps
            ):
                self.final_step = self.current_step + 1

            policy = (
                self.speeding_policy
                if self.use_speeding_policy
                else self.epsilon_greedy_policy
            )
            action = self._select_action(policy(state, available_actions(state)))
            self.actions[self._access_index(self.current_step + 1)] = action
        else:
            action = Action(0, 0)

        update_step = self.current_step - self.step_no + 1
        if update_step >= 0:
            if self.use_importance_sampling:
                return_value_weight = self._return_value_weight(update_step)
            else:
                return_value_weight = 1.0

            return_value = self._return_value(update_step)
            state_t = self.states[self._access_index(update_step)]
            action_t = self.actions[self._access_index(update_step)]
            # TODO: Tutaj trzeba zaktualizować tablicę wartościującą akcje Q
            self.q[state_t, action_t] += (
                self.step_size
                * return_value_weight
                * (return_value - self.q[state_t, action_t])
            )

        if update_step == self.final_step - 1:
            self.finished = True

        self.current_step += 1
        return action

    def _return_value(self, update_step):
        return_value = 0.0
        # TODO: Tutaj trzeba policzyć zwrot G
        for i in range(
            update_step + 1, min(update_step + self.step_no, self.final_step) + 1
        ):
            return_value += (
                self.discount_factor ** (i - update_step - 1)
            ) * self.rewards[self._access_index(i)]

        if update_step + self.step_no < self.final_step:
            state = self.states[self._access_index(update_step + self.step_no)]
            action = self.actions[self._access_index(update_step + self.step_no)]
            return_value += (self.discount_factor**self.step_no) * self.q[state, action]

        return return_value

    def _return_value_weight(self, update_step):
        return_value_weight = 1.0
        # TODO: Tutaj trzeba policzyć korektę na różne prawdopodobieństwa ρ (ponieważ uczymy poza-polityką)
        for i in range(
            update_step + 1, min(update_step + self.step_no, self.final_step - 1) + 1
        ):
            state = self.states[self._access_index(i)]
            behpolicy = (
                self.speeding_policy
                if self.use_speeding_policy
                else self.epsilon_greedy_policy
            )
            behaviour_policy = behpolicy(state, available_actions(state))
            target_policy = self.greedy_policy(state, available_actions(state))
            action = self.actions[self._access_index(i)]
            return_value_weight *= target_policy[action] / behaviour_policy[action]

        return return_value_weight

    def finished_learning(self) -> bool:
        return self.finished

    def _access_index(self, index: int) -> int:
        return index % (self.step_no + 1)

    @staticmethod
    def _select_action(actions_distribution: dict[Action, float]) -> Action:
        actions = list(actions_distribution.keys())
        probabilities = list(actions_distribution.values())
        i = np.random.choice(list(range(len(actions))), p=probabilities)
        return actions[i]

    def epsilon_greedy_policy(
        self, state: State, actions: list[Action]
    ) -> dict[Action, float]:
        # TODO: tutaj trzeba ustalic prawdopodobieństwa wyboru akcji według polityki ε-zachłannej
        greedy = self._greedy_probabilities(state, actions)
        random_policy = self._random_probabilities(actions)

        probabilities = (
            1 - self.experiment_rate
        ) * greedy + random_policy * self.experiment_rate
        return {
            action: probability for action, probability in zip(actions, probabilities)
        }

    def speeding_policy(
        self, state: State, actions: list[Action]
    ) -> dict[Action, float]:
        greedy = self._greedy_probabilities(state, actions)
        random_policy = self._random_probabilities(actions)
        speeding = self._speeding_probabilities(actions)

        probabilities = (
            (1 - self.experiment_rate - self.speeding_rate) * greedy
            + self.experiment_rate * random_policy
            + self.speeding_rate * speeding
        )

        probabilities = self._normalise(probabilities)

        return {
            action: probability for action, probability in zip(actions, probabilities)
        }

    def greedy_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        probabilities = self._greedy_probabilities(state, actions)
        return {
            action: probability for action, probability in zip(actions, probabilities)
        }

    def _greedy_probabilities(self, state: State, actions: list[Action]) -> np.ndarray:
        values = np.array([self.q[state, action] for action in actions])
        maximal_spots = (values == np.max(values)).astype(float)
        return self._normalise(maximal_spots)

    def _speeding_probabilities(self, actions: list[Action]) -> np.ndarray:
        accelerations_x = np.array([action.a_x for action in actions])
        max_acceleration_x = np.max(accelerations_x)

        speeding_spots = (accelerations_x == max_acceleration_x).astype(float)

        return self._normalise(speeding_spots)

    @staticmethod
    def _random_probabilities(actions: list[Action]) -> np.ndarray:
        maximal_spots = np.array([1.0 for _ in actions])
        return OffPolicyNStepSarsaDriver._normalise(maximal_spots)

    @staticmethod
    def _normalise(probabilities: np.ndarray) -> np.ndarray:
        return skl_preprocessing.normalize(probabilities.reshape(1, -1), norm="l1")[0]

    def eval_mode(self) -> None:
        self.backup_experiment_rate = self.experiment_rate
        self.backup_step_size = self.step_size
        self.backup_speeding_rate = self.speeding_rate
        self.backup_use_speeding_policy = self.use_speeding_policy

        self.experiment_rate = 0.0
        self.speeding_rate = 0.0
        self.step_size = 0.0
        self.use_speeding_policy = False

    def train_mode(self) -> None:
        self.experiment_rate = self.backup_experiment_rate
        self.step_size = self.backup_step_size
        self.speeding_rate = self.backup_speeding_rate
        self.use_speeding_policy = self.backup_use_speeding_policy

    def save(self, path: str) -> None:
        path = Path(path)

        data = {
            "step_size": self.step_size,
            "step_no": self.step_no,
            "experiment_rate": self.experiment_rate,
            "discount_factor": self.discount_factor,
            "q": dict(self.q),
            "max_learning_steps": self.max_learning_steps,
            "use_speeding_policy": self.use_speeding_policy,
            "speeding_rate": self.speeding_rate,
            "use_importance_sampling": self.use_importance_sampling,
        }

        with path.open("wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "OffPolicyNStepSarsaDriver":
        path = Path(path)

        with path.open("rb") as f:
            data = pickle.load(f)

        driver = cls(
            step_size=data["step_size"],
            step_no=data["step_no"],
            experiment_rate=data["experiment_rate"],
            discount_factor=data["discount_factor"],
            max_learning_steps=data["max_learning_steps"],
            use_speeding_policy=data.get("use_speeding_policy", False),
            speeding_rate=data.get("speeding_rate", 0.2),
            use_importance_sampling=data.get("use_importance_sampling", True),
        )

        driver.q = collections.defaultdict(float, data["q"])

        return driver
