from abc import ABC, abstractmethod
from pathlib import Path


class DataHandler(ABC):
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
    ):
        self.input_path = input_path
        self.output_path = output_path

    @abstractmethod
    def read_image(self, filename):
        pass

    @abstractmethod
    def save_macthes(self, matches):
        pass
