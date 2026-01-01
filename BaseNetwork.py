import numpy as np
from typing import Any, Optional

class BaseNetwork:
    """
    Base class for all Hopfield networks.
    """
    def __init__(self, size):
        """
        Initialize the network.
        """
        pass

    def train(self, patterns):
        raise NotImplementedError("Subclasses must implement train method")

    def energy(self, state):
        raise NotImplementedError("Subclasses must implement energy method")

    def get_field_map(self, state):
        raise NotImplementedError("Subclasses must implement get_field_map method")

    def restore_memory(self, input_state: np.ndarray, original_state: np.ndarray, steps: Optional[int] = None) -> Any:
        raise NotImplementedError("Subclasses must implement restore_memory method")