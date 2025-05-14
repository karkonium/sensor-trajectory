import numpy as np
from relfection import reflect_data_y
from rotation import rotate_data_90
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Test/visulaization 
    data = np.array([[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ]])

    print("Original Data:")
    print(data)

    data_reflected = reflect_data_y(data)
    print("\nReflected Data (about y-mid domain):")
    print(data_reflected)

    data_rotated = rotate_data_90(data)
    print("\nRotated Data (90° counterclockwise):")
    print(data_rotated)

    fig, axes = plt.subplots(1, 3)
    im = axes[0].imshow(data.T, origin='lower', extent=[0, 3, 0, 4], cmap='viridis')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    im = axes[1].imshow(data_reflected.T, origin='lower', extent=[0, 3, 0, 4], cmap='viridis')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    im = axes[2].imshow(data_rotated.T, origin='lower', extent=[0, 4, 0, 3], cmap='viridis')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    data.reshape(1, 4 * 3)