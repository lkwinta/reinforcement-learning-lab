from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter
import json


class BaseLogger(ABC):
    @abstractmethod
    def __enter__(self): ...

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_traceback): ...

    @abstractmethod
    def log_msg(self, msg: str) -> None: ...

    @abstractmethod
    def log_scalar(
        self, name: str, value: float, step: Optional[int] = None
    ) -> None: ...


class SilentLogger(BaseLogger):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def log_msg(self, msg: str) -> None:
        pass

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        pass


class TensorboardLogger(BaseLogger):
    def __init__(self, save_dir: Optional[Path] = None) -> None:
        super().__init__()
        self.save_dir = save_dir

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def log_msg(self, msg: str) -> None:
        print(msg)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        self.writer.add_scalar(name, value, step)

    def open(self) -> None:
        self.writer = SummaryWriter(log_dir=self.save_dir)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


class JSONLogger(BaseLogger):
    def __init__(self, save_path: Path, save_every: Optional[int] = None) -> None:
        super().__init__()
        self.save_path = save_path
        self.data = []
        self.save_every = save_every
        self.counter = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.save()

    def log_msg(self, msg: str) -> None:
        print(msg)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        entry = {"name": name, "value": value}
        if step is not None:
            entry["step"] = step
        self.data.append(entry)
        if self.save_every is not None:
            self.counter += 1
            if self.counter >= self.save_every:
                self.save()
                self.counter = 0

    def save(self) -> None:
        with open(self.save_path, "w") as f:
            json.dump(self.data, f, indent=4)
