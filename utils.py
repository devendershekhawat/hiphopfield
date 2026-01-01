import numpy as np

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

def break_art_and_flattern(art: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Takes a 32x32 art and breaks it in half"""
    # We take the Invader and WIPE OUT the right half
    input_broken = art.copy()
    reshaped = input_broken.reshape(SIZE, SIZE)
    reshaped[:, SIZE//2:] = -1 # Erase right half (set to background)
    input_broken = reshaped.flatten()
    return input_broken

def softmax(x: np.ndarray) -> np.ndarray:
  """Numerically stable softmax"""
  # Subtract max to prevent overflow
  x_shifted = x - np.max(x)
  exp_x = np.exp(x_shifted)
  return exp_x / np.sum(exp_x)