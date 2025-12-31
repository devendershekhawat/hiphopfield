import numpy as np

SIZE = 32

def get_accuracy(original, restored):
  return np.sum(original == restored) / len(original)

def break_art_and_flattern(art, size=SIZE):
    """Takes a 32x32 art and breaks it in half"""
    # We take the Invader and WIPE OUT the right half
    input_broken = art.copy()
    reshaped = input_broken.reshape(SIZE, SIZE)
    reshaped[:, SIZE//2:] = -1 # Erase right half (set to background)
    input_broken = reshaped.flatten()
    return input_broken