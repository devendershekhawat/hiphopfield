import numpy as np
from utils import softmax
from BaseNetwork import BaseNetwork
import matplotlib.pyplot as plt
from typing import Optional
from ModernHopfieldRestoration import ModernHopfieldRestoration

class ModernHopfieldNetwork(BaseNetwork):
    """ Creates a Modern Hopefield Network that can train on sizeXsize images"""
    def __init__(self, size: int = 32, beta: float = 10.0, binary: bool = True):
        self.size: int = size
        self.n: int = size * size
        self.memory_matrix: np.ndarray | None = None # (self.n, m) matrix of memories where m is the number of memories
        self.beta: float = beta # Beta is the temperature of the system
        self.binary: bool = binary # Whether to use binary or continuous values
        self.normalize: bool = False # Whether to normalize the patterns before training
        
    def train(self, patterns: list[np.ndarray]) -> np.ndarray:
        print(f"Learning {len(patterns)} memories with {self.n} neurons...")
        print(f"Synaptic Connections: {self.n**2:,} weights.")
        if self.memory_matrix is None:
            self.memory_matrix = np.zeros((len(patterns), self.n))
        print(f"Memory matrix shape: {self.memory_matrix.shape}")
        for i, pattern in enumerate(patterns):
            self.memory_matrix[i] = pattern
        return self.memory_matrix

    def train_normalized(self, patterns: list[np.ndarray]) -> np.ndarray:
        print(f"Learning {len(patterns)} memories with {self.n} neurons...")
        self.normalize = True
        if self.memory_matrix is None:
            self.memory_matrix = np.zeros((len(patterns), self.n))
            
        for i, pattern in enumerate(patterns):
            norm = np.linalg.norm(pattern)
            if norm > 0:
                self.memory_matrix[i] = pattern / norm
            else:
                self.memory_matrix[i] = pattern
                
        print(f"Memory matrix shape: {self.memory_matrix.shape}")
        return self.memory_matrix

    def energy(self, state: np.ndarray) -> float:
        if self.memory_matrix is None:
            raise ValueError("Memory matrix is not set")

        similarities = np.dot(self.memory_matrix.T, state)
        
        weighted_inputs = self.beta * similarities
        max_val = np.max(weighted_inputs) 
        
        log_sum_exps = max_val + np.log(np.sum(np.exp(weighted_inputs - max_val)))

        energy = -(1 / self.beta) * log_sum_exps 
        
        return energy + 0.5 * np.sum(state**2)

    def restore_memory(self, input_state: np.ndarray, original_state: np.ndarray, steps: Optional[int] = None) -> ModernHopfieldRestoration:
      if self.memory_matrix is None:
        raise ValueError("Memory matrix is not set")
      restoration = ModernHopfieldRestoration(self.beta, self.memory_matrix, self.size, original_state)
      if self.normalize:
        input_state = input_state / np.linalg.norm(input_state)
      restoration.set_input(input_state)
      s = input_state.copy()
      similarities = np.dot(self.memory_matrix, s)
      attention_weights = softmax(self.beta * similarities)
      output_state = np.sign(np.dot(attention_weights, self.memory_matrix)) if self.binary else np.dot(attention_weights, self.memory_matrix)
      restoration.set_restored_output(output_state, self.normalize)
      return restoration
