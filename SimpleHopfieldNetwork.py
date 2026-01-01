import numpy as np
from utils import get_accuracy
from BaseNetwork import BaseNetwork
from SimpleHopefieldRestoration import SimpleHopefieldRestoration, State
import matplotlib.pyplot as plt
class SimpleHopfieldNetwork(BaseNetwork):
    """
    A simple NxN Hopfield network
    """
    def __init__(self, size):
        self.size = size
        self.n = size * size
        self.W = np.zeros((self.n, self.n))

    def train(self, patterns):
        print(f"Learning {len(patterns)} memories with {self.n} neurons...")
        print(f"Synaptic Connections: {self.n**2:,} weights.")
        for p in patterns:
            self.W += np.outer(p, p)
        np.fill_diagonal(self.W, 0)
        # We normalize the weights to keep the field values manageable
        self.W /= self.n 
        return self.W

    def energy(self, state):
        """
        Calculates the energy of the network for a given state
        """
        return -0.5 * np.dot(state, np.dot(self.W, state))

    def get_field_map(self, state):
        """Returns the 'Pressure' map (What the neurons want to do)"""
        return np.dot(self.W, state)

    def restore_memory(self, input_state, original_state, steps=50):
        restoration = SimpleHopefieldRestoration(self.W, original_state, input_state)
        for i in range(steps*self.n):
            idx = np.random.randint(self.n)
            new_value = restoration.states[-1].value.copy()
            field = np.dot(self.W[idx], new_value)
            new_value[idx] = 1 if field >= 0 else -1
            restoration.add_state(new_value)
        return restoration