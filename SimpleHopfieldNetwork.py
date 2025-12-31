import numpy as np
from utils import get_accuracy

class SimpleHopfieldNetwork:
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

    def energy(self, state):
        """
        Calculates the energy of the network for a given state
        """
        return -0.5 * np.dot(state, np.dot(self.W, state))

    def get_field_map(self, state):
        """Returns the 'Pressure' map (What the neurons want to do)"""
        return np.dot(self.W, state)

    def restore_memory(self, input_state, original_state, steps=50):
        """
        Restores a memory from a given input state
        """
        s = input_state.copy()
        
        # We save snapshots at specific intervals to make a 'movie'
        snapshots = [0, 30, int(steps*self.n/2), steps*self.n - 1]
        saved_states = []
        energies = []
        accuracies = []
        
        print("Dreaming restoration...")
        for i in range(steps * self.n):
            # Asynchronous Update
            idx = np.random.randint(self.n)
            field = np.dot(self.W[idx], s)
            s[idx] = 1 if field >= 0 else -1
            accuracy = get_accuracy(original_state, s)
            accuracies.append(accuracy)
            
            # Record Energy
            if i % 500 == 0:
                energies.append(self.energy(s))
            
            # Save Snapshot if it's time
            if i in snapshots or i == 0:
                saved_states.append(s.copy())
        return s, saved_states, energies, accuracies