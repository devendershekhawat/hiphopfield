import numpy as np
import matplotlib.pyplot as plt

def create_arts(size=32):
    """
    Generates 10 distinct, recognizable binary icons of size x size.
    Returns a list of flattened arrays (size*size,).
    Pixels are +1 (ON) and -1 (OFF).
    """
    arts = []
    names = []
    
    # Helper: Upscales a small sprite to the target size
    # logical_grid: The small pixel art (e.g., 8x8)
    # returns: The 32x32 version
    def render_sprite(logical_grid):
        logical_grid = np.array(logical_grid)
        current_h, current_w = logical_grid.shape
        scale_h = size // current_h
        scale_w = size // current_w
        
        # Kronecker product acts like a "zoom" for pixel art
        img = np.kron(logical_grid, np.ones((scale_h, scale_w)))
        
        # Center crop/pad if exact size mismatch occurs
        final = np.ones((size, size)) * -1
        # Insert the image into the center
        start_r = (size - img.shape[0]) // 2
        start_c = (size - img.shape[1]) // 2
        final[start_r:start_r+img.shape[0], start_c:start_c+img.shape[1]] = np.where(img > 0, 1, -1)
        return final.flatten()

    # 1. Pixel Heart
    # 0=background, 1=foreground
    heart = [
        [0,0,0,0,0,0,0,0],
        [0,1,1,0,0,1,1,0],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [0,1,1,1,1,1,1,0],
        [0,0,1,1,1,1,0,0],
        [0,0,0,1,1,0,0,0]
    ]
    arts.append((render_sprite(heart), "Heart"))
    names.append("Heart")

    # 2. Space Invader (Classic)
    invader = [
        [0,0,1,0,0,0,0,0,1,0,0],
        [0,0,0,1,0,0,0,1,0,0,0],
        [0,0,1,1,1,1,1,1,1,0,0],
        [0,1,1,0,1,1,1,0,1,1,0],
        [1,1,1,1,1,1,1,1,1,1,1],
        [1,0,1,1,1,1,1,1,1,0,1],
        [1,0,1,0,0,0,0,0,1,0,1],
        [0,0,0,1,1,0,1,1,0,0,0]
    ]
    arts.append((render_sprite(invader), "Invader"))
    names.append("Invader")
    # 3. Pac-Man (Waka Waka)
    pacman = [
        [0,0,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,0,0], # Mouth open
        [1,1,1,1,0,0,0,0],
        [1,1,1,1,1,1,0,0],
        [0,1,1,1,1,1,1,1],
        [0,0,1,1,1,1,1,0]
    ]
    arts.append((render_sprite(pacman), "Pacman"))
    names.append("Pacman")
    # 4. The Ghost (Pacman Enemy)
    ghost = [
        [0,0,0,1,1,1,0,0,0],
        [0,0,1,1,1,1,1,0,0],
        [0,1,1,1,1,1,1,1,0],
        [1,1,0,1,1,1,0,1,1], # Eyes
        [1,1,0,1,1,1,0,1,1],
        [1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1],
        [1,0,1,0,1,0,1,0,1] # Wavy bottom
    ]
    arts.append((render_sprite(ghost), "Ghost"))
    names.append("Ghost")
    # 5. Yin Yang (Approximation)
    yin_yang = [
        [0,0,1,1,1,1,0,0],
        [0,1,1,1,1,1,1,0],
        [1,1,1,0,0,1,1,1],
        [1,1,0,0,0,0,1,1],
        [1,1,1,1,1,0,1,1],
        [1,1,1,1,0,0,1,1],
        [0,1,1,0,0,1,1,0],
        [0,0,0,0,1,1,0,0]
    ]
    arts.append((render_sprite(yin_yang), "Yin Yang"))
    names.append("Yin Yang")
    # 6. Mushroom (Mario Style)
    mushroom = [
        [0,0,0,1,1,1,1,0,0,0],
        [0,0,1,1,1,1,1,1,0,0],
        [0,1,1,0,0,1,1,0,1,0], # Spots
        [1,1,1,1,1,1,1,1,1,1],
        [1,0,1,1,1,1,1,1,0,1],
        [0,0,1,1,1,1,1,1,0,0],
        [0,0,0,1,1,1,1,0,0,0]
    ]
    arts.append((render_sprite(mushroom), "Mushroom"))
    names.append("Mushroom")
    # 7. Skull
    skull = [
        [0,0,1,1,1,1,0,0],
        [0,1,1,1,1,1,1,0],
        [1,1,0,1,1,0,1,1], # Eyes
        [1,1,1,1,1,1,1,1],
        [0,1,1,0,0,1,1,0], # Nose area
        [0,1,0,1,1,0,1,0], # Teeth
        [0,0,0,0,0,0,0,0]
    ]
    arts.append((render_sprite(skull), "Skull"))
    names.append("Skull")
    # 8. Tree (Pine)
    tree = [
        [0,0,0,0,1,0,0,0,0],
        [0,0,0,1,1,1,0,0,0],
        [0,0,1,1,1,1,1,0,0],
        [0,1,1,1,1,1,1,1,0],
        [0,0,0,1,1,1,0,0,0], # Second tier
        [0,0,1,1,1,1,1,0,0],
        [0,1,1,1,1,1,1,1,0],
        [0,0,0,0,1,0,0,0,0]  # Trunk
    ]
    arts.append((render_sprite(tree), "Tree"))
    names.append("Tree")
    # 9. Creeper Face
    creeper = [
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,0,0,1,1,0,0,1], # Eyes
        [1,0,0,1,1,0,0,1],
        [1,1,1,0,0,1,1,1], # Nose
        [1,1,0,0,0,0,1,1], # Mouth
        [1,1,0,0,0,0,1,1],
        [1,1,1,1,1,1,1,1]
    ]
    arts.append((render_sprite(creeper), "Creeper"))
    names.append("Creeper")
    # 10. The Sword
    sword = [
        [0,0,0,0,1,0,0,0,0],
        [0,0,0,1,1,1,0,0,0],
        [0,0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0,0],
        [0,1,1,1,1,1,1,1,0], # Hilt
        [0,0,0,0,1,0,0,0,0], # Handle
        [0,0,0,0,1,0,0,0,0]
    ]
    arts.append((render_sprite(sword), "Sword"))
    names.append("Sword")
    return arts, names

def get_random_binary_images(size=32, n=5):
    """
    Generate n random binary images of size `size` x `size`,
    with each pixel being either -1 or 1.
    
    Returns:
        images: numpy array of shape (n, size, size)
    """
    images = np.random.choice([-1, 1], size=(n, size, size))
    flattened_images = [image.flatten() for image in images]
    return flattened_images