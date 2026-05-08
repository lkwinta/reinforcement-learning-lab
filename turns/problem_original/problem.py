from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
import itertools
import random
from typing import NamedTuple, Optional, Protocol

import matplotlib.image as mpl_image
import numpy as np
from tqdm import tqdm

import utils

MIN_VX = 0
MAX_VX = 3
MIN_VY = -3
MAX_VY = 3


class Position(NamedTuple):
    x: int
    y: int


class Action(NamedTuple):
    a_x: int
    a_y: int


class State(NamedTuple):
    x: int
    y: int
    v_x: int
    v_y: int


def available_actions(state: State) -> list[Action]:
    return [
        action for action in Car.POTENTIAL_ACTIONS if (
                MIN_VX <= state.v_x + action.a_x <= MAX_VX and
                MIN_VY <= state.v_y + action.a_y <= MAX_VY and
                (state.v_x + action.a_x != 0 or state.v_y + action.a_y != 0)
        )
    ]


class Corner:
    def __init__(self, name: str) -> None:
        self.image: np.ndarray = mpl_image.imread(f'corners/{name}.png')
        self.track: np.ndarray = np.flip(self.image[:, :, 0] + self.image[:, :, 1], 0)
        self.track[self.track == 2.0] = 1.0
        self.starting_positions: set[Position] = self._determine_positions(
            np.flip(self.image[:, :, 1] - self.image[:, :, 2], 0)
        )
        self.terminal_positions: set[Position] = self._determine_positions(
            np.flip(self.image[:, :, 0] - self.image[:, :, 2], 0)
        )
        self.image = np.flip(self.image, 0)

    def contains(self, position: Position) -> bool:
        return (0 < position.x < self.track.shape[0] and
                0 < position.y < self.track.shape[1] and
                self.track[position] == 1.0)

    @staticmethod
    def _determine_positions(image: np.ndarray) -> set[Position]:
        return set(
            Position(x, y) for (x, y), value in np.ndenumerate(image) if value == 1.0
        )


class Car:
    POTENTIAL_ACTIONS: list[Action] = [Action(ax, ay) for ax, ay in itertools.product((-1, 0, 1), (-1, 0, 1))]

    def __init__(self, position: Position, driver: Driver, environment: Environment):
        self.x: int = position.x
        self.y: int = position.y
        self.v_x: int = 0
        self.v_y: int = 0
        self.driver: Driver = driver
        self.environment: Environment = environment
        self.total_penalties: int = 0
        self.last_penalty: Optional[int] = None

    def state(self) -> State:
        return State(self.x, self.y, self.v_x, self.v_y)

    def position(self) -> Position:
        return Position(self.x, self.y)

    def next_position(self) -> Position:
        if self.position() in self.environment.corner.terminal_positions:
            return self.position()
        else:
            return Position(self.x + self.v_x, self.y + self.v_y)

    def drive(self):
        if self.last_penalty is not None:
            action = self.driver.control(self.state(), self.last_penalty)
        else:
            action = self.driver.start_attempt(self.state())
        if self.last_penalty == 0:
            self.v_x, self.v_y = 0, 0
            action = Action(0, 0)
        self.last_penalty = self.environment.time_step(self, action)
        self.total_penalties += self.last_penalty


class Driver(Protocol):
    @abstractmethod
    def start_attempt(self, state: State) -> Action:
        raise NotImplementedError

    @abstractmethod
    def control(self, state: State, last_reward: int) -> Action:
        raise NotImplementedError

    @abstractmethod
    def finished_learning(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def eval_mode(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def train_mode(self) -> None:
        raise NotImplementedError


@dataclass
class Environment:
    corner: Corner
    steering_fail_chance: float

    def spawn_car(self, driver: Driver) -> Car:
        return Car(self._random_start(), driver, self)

    def time_step(self, car: Car, action: Action) -> int:
        action = Action(0, 0) if random.random() < self.steering_fail_chance else action
        car.v_x, car.v_y = car.v_x + action.a_x, car.v_y + action.a_y
        next_position = car.next_position()
        if not self.corner.contains(next_position):
            next_position = self._random_start()
            car.v_x, car.v_y = 0, 0
        car.x, car.y = next_position.x, next_position.y
        return 0 if next_position in self.corner.terminal_positions else -1

    def _random_start(self) -> Position:
        return random.sample(list(self.corner.starting_positions), 1)[0]


@dataclass
class Experiment:
    environment: Environment
    driver: Driver
    number_of_episodes: int
    current_episode_no: int = 0
    penalties: Optional[list] = None  # tutaj będą się gromadzić kary przyznane w kolejnych epizodach,
    plots_path: str = 'plots'
    drawing_frequency: int = 50
    averaging_window_size: int = 25

    eval_active: bool = False
    eval_frequency: int = 100
    eval_episodes: int = 10
    eval_penalties: Optional[list] = None

    save_driver: bool = False
    show_progress: bool = True

    def load_inital_driver(self, cls, path: str) -> None:
        self.driver = cls.load(path)

    def run(self) -> None:
        self.penalties = []

        iterator = range(self.number_of_episodes)
        if self.show_progress:
            iterator = tqdm(iterator)

        for _ in iterator:
            episode_penalty = self._run_episode()[0]
            self.penalties.append(episode_penalty)
            self.current_episode_no += 1

    def _run_episode(self) -> int:
        total_penalty, positions = self._episode()
        self._draw_episode(positions)
        if self.eval_active:
            self._evaluate_driver()
        return total_penalty, positions
    
    def _episode(self) -> tuple[int, list[Position]]:
        positions = []
        car = self.environment.spawn_car(self.driver)
        while True:
            positions.append(car.position())
            car.drive()
            if car.driver.finished_learning():
                positions.append(car.position())
                break

        return car.total_penalties, positions

    def _draw_episode(self, positions: list[Position]) -> None:
        if self.current_episode_no % self.drawing_frequency == 0:
            utils.draw_episode(self.environment.corner.image, positions, self.current_episode_no, self.plots_path)
            utils.draw_penalties_plot(self.penalties, self.averaging_window_size, self.current_episode_no, self.plots_path)

    def _evaluate_driver(self) -> None:
        if self.current_episode_no % self.eval_frequency == 0:
            
            self.driver.eval_mode()
            
            eval_penalties = []
            for _ in range(self.eval_episodes):
                eval_penalties.append(self._episode()[0])
            avg_eval_penalty = sum(eval_penalties) / len(eval_penalties)

            if self.eval_penalties is None:
                self.eval_penalties = []

            self.eval_penalties.append(avg_eval_penalty)

            self.driver.train_mode()
