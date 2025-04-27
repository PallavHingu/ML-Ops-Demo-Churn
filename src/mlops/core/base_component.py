from abc import ABC, abstractmethod
import logging, yaml, os
from .logger import get_logger

class BaseComponent(ABC):
    """Abstract parent: every domain class derives from this."""

    def __init__(self):
        self.log = get_logger(self.__class__.__name__)
        # with open(config_path, "r") as f:
        #     self.cfg = yaml.safe_load(f)

    @abstractmethod
    def run(self, *args, **kwargs):
        """Execute component functionality."""