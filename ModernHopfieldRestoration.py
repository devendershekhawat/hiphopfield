import numpy as np
from utils import softmax, get_mean_error
import matplotlib.pyplot as plt

class ModernHopfieldRestoration:
  def __init__(self, beta: float = 10.0, memory_matrix: np.ndarray | None = None, size: int = 32, original_state: np.ndarray | None = None):
    self.size: int = size
    self.initial_energy: float | None = None
    self.final_energy: float | None = None
    self.mean_error = None
    self.input: np.ndarray | None = None
    self.restored_output: np.ndarray | None = None
    self.memory_matrix: np.ndarray | None = memory_matrix
    self.beta: float = beta
    self.original_state: np.ndarray | None = original_state
    self.similarities: np.ndarray | None = None
    self.attention_weights: np.ndarray | None = None

  def set_input(self, input: np.ndarray):
    self.input = input
    self.initial_energy = self.calculate_energy(input, self.beta)

  def set_restored_output(self, restored_output: np.ndarray, normalized: bool = False):
    if self.original_state is None:
      raise ValueError("Original state is not set")
    self.restored_output = restored_output
    self.final_energy = self.calculate_energy(restored_output, self.beta)
    self.mean_error = get_mean_error(self.original_state, self.restored_output, normalized)

  def set_similarities(self, similarities: np.ndarray):
    self.similarities = similarities
    self.attention_weights = softmax(self.beta * similarities)

  def calculate_energy(self, input_state: np.ndarray, beta: float = 10.0) -> float:
    if self.memory_matrix is None:
      raise ValueError("Memory matrix is not set")
    similarities = np.dot(self.memory_matrix, input_state)
    weighted_inputs = beta * similarities
    max_val = np.max(weighted_inputs)
    log_sum_exps = max_val + np.log(np.sum(np.exp(weighted_inputs - max_val)))
    energy = -(1 / beta) * log_sum_exps + 0.5 * np.sum(input_state**2)
    return energy

  def plot_restoration(self):
    if self.original_state is None or self.input is None or self.restored_output is None:
      raise ValueError("Input or restored output is not set")
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax1.set_title("Original Image")
    ax2.set_title("Damaged Image")
    ax3.set_title("Restored Image")

    ax1.imshow(self.original_state.reshape(self.size, self.size), cmap="plasma")
    ax2.imshow(self.input.reshape(self.size, self.size), cmap="plasma")
    ax3.imshow(self.restored_output.reshape(self.size, self.size), cmap="plasma")

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    ax4.text(0.5, 0.5, f"Initial Energy:\n{self.initial_energy:.2f}", 
             ha='center', va='center', fontsize=24, weight='bold',
             transform=ax4.transAxes)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    ax5.text(0.5, 0.5, f"Final Energy:\n{self.final_energy:.2f}", 
             ha='center', va='center', fontsize=24, weight='bold',
             transform=ax5.transAxes)
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    ax6.text(0.5, 0.5, f"Mean Error:\n{self.mean_error:.4f}", 
             ha='center', va='center', fontsize=24, weight='bold',
             transform=ax6.transAxes)

    plt.show()