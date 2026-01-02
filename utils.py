import numpy as np
import matplotlib.pyplot as plt

from ModernHopFieldNetwork import ModernHopfieldNetwork
import SimpleHopfieldNetwork

SIZE = 32

def get_mean_error(original: np.ndarray, restored: np.ndarray, normalized: bool = False):
    if normalized:
      original_norm = np.linalg.norm(original)
      original = original / original_norm
    matching = original == restored
    mean_error = np.mean(np.abs(original - restored))
    return mean_error * 100

def get_accuracy(original: np.ndarray, restored: np.ndarray, normalized: bool = False):
    if normalized:
      original_norm = np.linalg.norm(original)
      original = original / original_norm
    matching = original == restored
    return np.sum(matching) / len(original)

def softmax(x: np.ndarray) -> np.ndarray:
  """Numerically stable softmax"""
  # Subtract max to prevent overflow
  x_shifted = x - np.max(x)
  exp_x = np.exp(x_shifted)
  return exp_x / np.sum(exp_x)

def show_img(ax, flat_vec):
    ax.imshow(flat_vec.reshape(SIZE, SIZE), cmap='plasma', vmin=-1, vmax=1)
    ax.axis('off')

def plot_selected_arts(selected_arts, names=None):
  n = len(selected_arts)
  cols = min(n, 5)
  rows = (n + cols - 1) // cols
  fig, axs = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
  fig.suptitle("Selected Memories to train hopfield network", fontsize=16)
  for i in range(rows * cols):
      ax = axs.flat[i] if n > 1 else axs
      if i < n:
          show_img(ax, selected_arts[i])
          ax.set_title(f"{names[i] if names else f"Memory {i + 1}"}")
      else:
          ax.axis('off')
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.show()

def break_art_and_flattern(art, size=SIZE):
    """Takes a 32x32 art and breaks it in half"""
    # We take the Invader and WIPE OUT the right half
    input_broken = art.copy()
    reshaped = input_broken.reshape(SIZE, SIZE)
    reshaped[:, SIZE//2:] = 0 
    input_broken = reshaped.flatten()
    return input_broken

def break_continuous_art(art, size=128, damage_type="alternating_right"):
    """
    Corrupts a continuous (grayscale) image.
    args:
        art: Flattened numpy array of the image
        size: Width/Height of the image
        damage_type: 'mask_right', 'mask_center', 'static', or 'alternating_right'
        
        - 'mask_right': zero out right half
        - 'mask_center': black out center square
        - 'static': add gaussian noise
        - 'alternating_right': alternate rows on right half set to 0
    """
    # 1. Reshape to 2D so we can manipulate regions
    input_broken = art.copy().reshape(size, size)
    
    if damage_type == "mask_right":
        # WIPE OUT the right half (Set to 0.0 or -1.0 depending on your data range)
        # We use 0.0 (Gray) here as a "neutral" missing value
        input_broken[:, size//2:] = 0.0 
        
    elif damage_type == "mask_center":
        # Put a black box in the middle
        margin = size // 4
        input_broken[margin:-margin, margin:-margin] = 0.0
        
    elif damage_type == "static":
        # Add Gaussian Noise (The most common continuous test)
        noise = np.random.normal(0, 0.4, (size, size)) # Mean 0, Sigma 0.4
        input_broken = input_broken + noise
        # Clip to keep values valid (e.g., between -1 and 1)
        input_broken = np.clip(input_broken, -1.0, 1.0)

    elif damage_type == "alternating_right":
        # Left half stays the same, right half: set alternate rows to 0
        # Loop through each row: for even rows (or odd, your choice), set right half to 0
        for row in range(size):
            if row % 2 == 0:
                input_broken[row, size//2:] = 0.0

    # 2. Flatten back to vector
    return input_broken.flatten()

def attempt_restore_memory(
        selected_arts,
        nameorindex,
        steps=50,
        names=[],
        net: ModernHopfieldNetwork | SimpleHopfieldNetwork.SimpleHopfieldNetwork | None = None,
        continuous=False,
        damage_type="alternating_right"
    ):
    if net is None:
        net = ModernHopfieldNetwork(SIZE)
        net.train(selected_arts)
    result = {}
    if isinstance(nameorindex, int):
        art = selected_arts[nameorindex]
    else:
        art = selected_arts[names.index(nameorindex)]
        name = nameorindex
        result["name"] = name
    broken_and_flattened = break_art_and_flattern(art) if not continuous else break_continuous_art(art, damage_type=damage_type)
    restoration = net.restore_memory(broken_and_flattened, art, steps=steps)
    restoration.plot_restoration()
    return restoration