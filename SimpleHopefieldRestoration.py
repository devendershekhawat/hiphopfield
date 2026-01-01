import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from utils import get_accuracy

class State:
  value: np.ndarray
  energy: float
  accuracy: float
  def __init__(self, value: np.ndarray, energy: float, accuracy: float):
    self.value = value
    self.energy = energy
    self.accuracy = accuracy

class SimpleHopefieldRestoration:
  def __init__(self, W: np.ndarray, original_state: np.ndarray, input_state: np.ndarray):
    self.W = W
    self.original_state = original_state
    self.states = []
    self.states.append(State(
      input_state.copy(),
      self.calculate_energy(input_state.copy()),
      get_accuracy(original_state, input_state.copy()))
    )

  def calculate_energy(self, state: np.ndarray):
    return -0.5 * np.dot(state, np.dot(self.W, state))

  def add_state(self, state: np.ndarray):
    self.states.append(State(
      state.copy(),
      self.calculate_energy(state.copy()),
      get_accuracy(self.original_state, state.copy()))
    )

  def plot_restoration(self):
    num_states = len(self.states)
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 4, height_ratios=[2, 2, 1])

    img_shape = int(np.sqrt(self.original_state.shape[0]))

    # Row 1: 4 images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Original Image")
    ax1.imshow(self.original_state.reshape(img_shape, img_shape), cmap="plasma")
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Input")
    ax2.imshow(self.states[0].value.reshape(img_shape, img_shape), cmap="plasma")
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    early_idx = int(num_states // 3)
    ax3.set_title("Early Recall")
    ax3.imshow(self.states[early_idx].value.reshape(img_shape, img_shape), cmap="plasma")
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    late_idx = int(2 * num_states // 3)
    ax4.set_title("Late Recall")
    ax4.imshow(self.states[late_idx].value.reshape(img_shape, img_shape), cmap="plasma")
    ax4.axis('off')

    # Row 2: Restored image, Final energy/accuracy text
    ax5 = fig.add_subplot(gs[1, 0:2])
    ax5.set_title("Restored Image")
    ax5.imshow(self.states[-1].value.reshape(img_shape, img_shape), cmap="plasma")
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 2:4])
    ax6.axis('off')
    final_energy = self.states[-1].energy
    final_accuracy = self.states[-1].accuracy
    ax6.text(0.5, 0.7, f"Final Energy:\n{final_energy:.2f}", ha='center', va='center', fontsize=26, weight='bold', color='darkblue', transform=ax6.transAxes)
    ax6.text(0.5, 0.3, f"Final Accuracy:\n{final_accuracy*100:.2f}%", ha='center', va='center', fontsize=26, weight='bold', color='darkgreen', transform=ax6.transAxes)

    # Row 3: Energies and Accuracies
    ax7 = fig.add_subplot(gs[2, 0:2])
    energies = [state.energy for state in self.states]
    ax7.plot(energies, marker='o', color='purple')
    ax7.set_title("Energy Across Iterations")
    ax7.set_xlabel("Iteration")
    ax7.set_ylabel("Energy")

    ax8 = fig.add_subplot(gs[2, 2:4])
    accuracies = [state.accuracy for state in self.states]
    ax8.plot([a * 100 for a in accuracies], marker='o', color='teal')
    ax8.set_title("Accuracy Across Iterations")
    ax8.set_xlabel("Iteration")
    ax8.set_ylabel("Accuracy (%)")

    plt.tight_layout()
    plt.show()