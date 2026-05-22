from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from numba import njit, types
from numba.typed import Dict

from .problem import (
    Action,
    Driver,
    State,
    POTENTIAL_ACTIONS_ARRAY,
    action_from_numba,
    state_to_numba,
    available_actions_jit,
)

ALMOST_INFINITE_STEP = 1_000_000

Q_KEY_TYPE = types.UniTuple(types.int64, 6)
Q_VALUE_TYPE = types.float64


def make_q_table():
    return Dict.empty(
        key_type=Q_KEY_TYPE,
        value_type=Q_VALUE_TYPE,
    )


@njit(cache=True)
def make_q_key(
    state: tuple[int, int, int, int],
    action: tuple[int, int],
) -> tuple[int, int, int, int, int, int]:
    return (
        state[0],
        state[1],
        state[2],
        state[3],
        action[0],
        action[1],
    )


@njit(cache=True)
def q_get(
    q,
    state: tuple[int, int, int, int],
    action: tuple[int, int],
) -> float:
    key = make_q_key(state, action)

    if key in q:
        return q[key]

    return 0.0


@njit(cache=True)
def q_update(
    q,
    state: tuple[int, int, int, int],
    action: tuple[int, int],
    step_size: float,
    return_value_weight: float,
    return_value: float,
) -> None:
    key = make_q_key(state, action)

    old_value = 0.0
    if key in q:
        old_value = q[key]

    q[key] = old_value + step_size * return_value_weight * (return_value - old_value)


@njit(cache=True)
def action_index(
    actions: np.ndarray,
    action: tuple[int, int],
) -> int:
    for i in range(actions.shape[0]):
        if actions[i, 0] == action[0] and actions[i, 1] == action[1]:
            return i

    return -1


@njit(cache=True)
def normalise_jit(probabilities: np.ndarray) -> np.ndarray:
    total = 0.0

    for i in range(probabilities.shape[0]):
        total += probabilities[i]

    result = np.empty(probabilities.shape[0], dtype=np.float64)

    if total == 0.0:
        uniform_probability = 1.0 / probabilities.shape[0]

        for i in range(probabilities.shape[0]):
            result[i] = uniform_probability

        return result

    for i in range(probabilities.shape[0]):
        result[i] = probabilities[i] / total

    return result


@njit(cache=True)
def random_probabilities_jit(actions: np.ndarray) -> np.ndarray:
    probabilities = np.empty(actions.shape[0], dtype=np.float64)
    probability = 1.0 / actions.shape[0]

    for i in range(actions.shape[0]):
        probabilities[i] = probability

    return probabilities


@njit(cache=True)
def greedy_probabilities_jit(
    q,
    state: tuple[int, int, int, int],
    actions: np.ndarray,
) -> np.ndarray:
    values = np.empty(actions.shape[0], dtype=np.float64)

    for i in range(actions.shape[0]):
        action = (actions[i, 0], actions[i, 1])
        values[i] = q_get(q, state, action)

    max_value = values[0]

    for i in range(1, values.shape[0]):
        if values[i] > max_value:
            max_value = values[i]

    probabilities = np.zeros(values.shape[0], dtype=np.float64)
    greedy_count = 0.0

    for i in range(values.shape[0]):
        if values[i] == max_value:
            probabilities[i] = 1.0
            greedy_count += 1.0

    for i in range(probabilities.shape[0]):
        probabilities[i] /= greedy_count

    return probabilities


@njit(cache=True)
def speeding_probabilities_jit(actions: np.ndarray) -> np.ndarray:
    max_acceleration_x = actions[0, 0]

    for i in range(1, actions.shape[0]):
        if actions[i, 0] > max_acceleration_x:
            max_acceleration_x = actions[i, 0]

    probabilities = np.zeros(actions.shape[0], dtype=np.float64)
    speeding_count = 0.0

    for i in range(actions.shape[0]):
        if actions[i, 0] == max_acceleration_x:
            probabilities[i] = 1.0
            speeding_count += 1.0

    for i in range(probabilities.shape[0]):
        probabilities[i] /= speeding_count

    return probabilities


@njit(cache=True)
def epsilon_greedy_probabilities_jit(
    q,
    state: tuple[int, int, int, int],
    actions: np.ndarray,
    experiment_rate: float,
) -> np.ndarray:
    greedy = greedy_probabilities_jit(q, state, actions)
    random_policy = random_probabilities_jit(actions)

    probabilities = np.empty(actions.shape[0], dtype=np.float64)

    for i in range(actions.shape[0]):
        probabilities[i] = (1.0 - experiment_rate) * greedy[
            i
        ] + experiment_rate * random_policy[i]

    return normalise_jit(probabilities)


@njit(cache=True)
def speeding_policy_probabilities_jit(
    q,
    state: tuple[int, int, int, int],
    actions: np.ndarray,
    experiment_rate: float,
    speeding_rate: float,
) -> np.ndarray:
    greedy = greedy_probabilities_jit(q, state, actions)
    random_policy = random_probabilities_jit(actions)
    speeding = speeding_probabilities_jit(actions)

    probabilities = np.empty(actions.shape[0], dtype=np.float64)

    greedy_rate = 1.0 - experiment_rate - speeding_rate

    for i in range(actions.shape[0]):
        probabilities[i] = (
            greedy_rate * greedy[i]
            + experiment_rate * random_policy[i]
            + speeding_rate * speeding[i]
        )

    return normalise_jit(probabilities)


@njit(cache=True)
def behaviour_policy_probabilities_jit(
    q,
    state: tuple[int, int, int, int],
    actions: np.ndarray,
    experiment_rate: float,
    use_speeding_policy: bool,
    speeding_rate: float,
) -> np.ndarray:
    if use_speeding_policy:
        return speeding_policy_probabilities_jit(
            q,
            state,
            actions,
            experiment_rate,
            speeding_rate,
        )

    return epsilon_greedy_probabilities_jit(
        q,
        state,
        actions,
        experiment_rate,
    )


@njit(cache=True)
def select_action_jit(
    actions: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[int, int]:
    sample = np.random.random()
    cumulative = 0.0

    for i in range(probabilities.shape[0]):
        cumulative += probabilities[i]

        if sample <= cumulative:
            return actions[i, 0], actions[i, 1]

    last = probabilities.shape[0] - 1
    return actions[last, 0], actions[last, 1]


@njit(cache=True)
def return_value_jit(
    q,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    update_step: int,
    step_no: int,
    final_step: int,
    discount_factor: float,
) -> float:
    return_value = 0.0

    end = update_step + step_no

    if final_step < end:
        end = final_step

    for i in range(update_step + 1, end + 1):
        index = i % (step_no + 1)
        power = i - update_step - 1
        return_value += (discount_factor**power) * rewards[index]

    if update_step + step_no < final_step:
        index = (update_step + step_no) % (step_no + 1)

        state = (
            states[index, 0],
            states[index, 1],
            states[index, 2],
            states[index, 3],
        )
        action = (
            actions[index, 0],
            actions[index, 1],
        )

        return_value += (discount_factor**step_no) * q_get(q, state, action)

    return return_value


@njit(cache=True)
def return_value_weight_jit(
    q,
    states: np.ndarray,
    actions_by_step: np.ndarray,
    update_step: int,
    step_no: int,
    final_step: int,
    experiment_rate: float,
    use_speeding_policy: bool,
    speeding_rate: float,
    use_importance_sampling: bool,
    potential_actions: np.ndarray,
) -> float:
    if not use_importance_sampling:
        return 1.0

    return_value_weight = 1.0

    end = update_step + step_no

    if final_step - 1 < end:
        end = final_step - 1

    for i in range(update_step + 1, end + 1):
        index = i % (step_no + 1)

        state = (
            states[index, 0],
            states[index, 1],
            states[index, 2],
            states[index, 3],
        )
        action = (
            actions_by_step[index, 0],
            actions_by_step[index, 1],
        )

        available_actions = available_actions_jit(state, potential_actions)

        behaviour_policy = behaviour_policy_probabilities_jit(
            q,
            state,
            available_actions,
            experiment_rate,
            use_speeding_policy,
            speeding_rate,
        )

        target_policy = greedy_probabilities_jit(
            q,
            state,
            available_actions,
        )

        selected_index = action_index(available_actions, action)

        if selected_index == -1:
            return 0.0

        behaviour_probability = behaviour_policy[selected_index]

        if behaviour_probability == 0.0:
            return 0.0

        return_value_weight *= target_policy[selected_index] / behaviour_probability

        if return_value_weight == 0.0:
            return 0.0

    return return_value_weight


@njit(cache=True)
def update_q_from_buffers_jit(
    q,
    states: np.ndarray,
    actions: np.ndarray,
    update_step: int,
    step_no: int,
    step_size: float,
    return_value_weight: float,
    return_value: float,
) -> None:
    index = update_step % (step_no + 1)

    state = (
        states[index, 0],
        states[index, 1],
        states[index, 2],
        states[index, 3],
    )
    action = (
        actions[index, 0],
        actions[index, 1],
    )

    q_update(
        q,
        state,
        action,
        step_size,
        return_value_weight,
        return_value,
    )


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
        if experiment_rate + speeding_rate > 1.0:
            raise ValueError("experiment_rate + speeding_rate must be <= 1.0")

        self.step_size: float = step_size
        self.step_no: int = step_no
        self.experiment_rate: float = experiment_rate
        self.discount_factor: float = discount_factor
        self.max_learning_steps: int = max_learning_steps

        self.use_speeding_policy: bool = use_speeding_policy
        self.speeding_rate: float = speeding_rate
        self.use_importance_sampling: bool = use_importance_sampling

        self.q = make_q_table()

        buffer_size = step_no + 1
        self.states = np.zeros((buffer_size, 4), dtype=np.int64)
        self.actions = np.zeros((buffer_size, 2), dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float64)

        self.current_step: int = 0
        self.final_step: int = ALMOST_INFINITE_STEP
        self.finished: bool = False

        self.backup_step_size = step_size
        self.backup_experiment_rate = experiment_rate
        self.backup_speeding_rate = speeding_rate
        self.backup_use_speeding_policy = use_speeding_policy

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        self.final_step = ALMOST_INFINITE_STEP
        self.finished = False

        index = self._access_index(self.current_step)
        self._store_state(index, state)

        action = self._select_action_for_state(state)
        self._store_action(index, action)

        return action

    def control(self, state: State, last_reward: int) -> Action:
        if self.current_step < self.final_step:
            index = self._access_index(self.current_step + 1)

            self.rewards[index] = float(last_reward)
            self._store_state(index, state)

            if self.final_step == ALMOST_INFINITE_STEP and (
                last_reward == 0 or self.current_step == self.max_learning_steps
            ):
                self.final_step = self.current_step + 1

            action = self._select_action_for_state(state)
            self._store_action(index, action)
        else:
            action = Action(0, 0)

        update_step = self.current_step - self.step_no + 1

        if update_step >= 0:
            return_value_weight = return_value_weight_jit(
                self.q,
                self.states,
                self.actions,
                update_step,
                self.step_no,
                self.final_step,
                self.experiment_rate,
                self.use_speeding_policy,
                self.speeding_rate,
                self.use_importance_sampling,
                POTENTIAL_ACTIONS_ARRAY,
            )

            return_value = return_value_jit(
                self.q,
                self.states,
                self.actions,
                self.rewards,
                update_step,
                self.step_no,
                self.final_step,
                self.discount_factor,
            )

            update_q_from_buffers_jit(
                self.q,
                self.states,
                self.actions,
                update_step,
                self.step_no,
                self.step_size,
                return_value_weight,
                return_value,
            )

        if update_step == self.final_step - 1:
            self.finished = True

        self.current_step += 1

        return action

    def finished_learning(self) -> bool:
        return self.finished

    def _access_index(self, index: int) -> int:
        return index % (self.step_no + 1)

    def _store_state(self, index: int, state: State) -> None:
        self.states[index, 0] = state.x
        self.states[index, 1] = state.y
        self.states[index, 2] = state.v_x
        self.states[index, 3] = state.v_y

    def _store_action(self, index: int, action: Action) -> None:
        self.actions[index, 0] = action.a_x
        self.actions[index, 1] = action.a_y

    def _select_action_for_state(self, state: State) -> Action:
        state_nb = state_to_numba(state)

        actions = available_actions_jit(
            state_nb,
            POTENTIAL_ACTIONS_ARRAY,
        )

        probabilities = behaviour_policy_probabilities_jit(
            self.q,
            state_nb,
            actions,
            self.experiment_rate,
            self.use_speeding_policy,
            self.speeding_rate,
        )

        action_nb = select_action_jit(actions, probabilities)

        return action_from_numba(action_nb)

    def epsilon_greedy_policy(
        self,
        state: State,
        actions: list[Action],
    ) -> dict[Action, float]:
        state_nb = state_to_numba(state)
        actions_array = np.array(
            [[action.a_x, action.a_y] for action in actions],
            dtype=np.int64,
        )

        probabilities = epsilon_greedy_probabilities_jit(
            self.q,
            state_nb,
            actions_array,
            self.experiment_rate,
        )

        return {
            action: float(probability)
            for action, probability in zip(actions, probabilities)
        }

    def speeding_policy(
        self,
        state: State,
        actions: list[Action],
    ) -> dict[Action, float]:
        state_nb = state_to_numba(state)
        actions_array = np.array(
            [[action.a_x, action.a_y] for action in actions],
            dtype=np.int64,
        )

        probabilities = speeding_policy_probabilities_jit(
            self.q,
            state_nb,
            actions_array,
            self.experiment_rate,
            self.speeding_rate,
        )

        return {
            action: float(probability)
            for action, probability in zip(actions, probabilities)
        }

    def greedy_policy(
        self,
        state: State,
        actions: list[Action],
    ) -> dict[Action, float]:
        state_nb = state_to_numba(state)
        actions_array = np.array(
            [[action.a_x, action.a_y] for action in actions],
            dtype=np.int64,
        )

        probabilities = greedy_probabilities_jit(
            self.q,
            state_nb,
            actions_array,
        )

        return {
            action: float(probability)
            for action, probability in zip(actions, probabilities)
        }

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
            "max_learning_steps": self.max_learning_steps,
            "use_speeding_policy": self.use_speeding_policy,
            "speeding_rate": self.speeding_rate,
            "use_importance_sampling": self.use_importance_sampling,
            "q": dict(self.q),
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

        for key, value in data["q"].items():
            driver.q[
                (
                    int(key[0]),
                    int(key[1]),
                    int(key[2]),
                    int(key[3]),
                    int(key[4]),
                    int(key[5]),
                )
            ] = float(value)

        return driver
