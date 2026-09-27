"""Run the noiseless point-pattern example; see docs/simulation.md."""

import math

import numpy as np

from image import generate_database_1, get_aerial_image
from main import find_position


def main():
    np.random.seed(0)
    database = generate_database_1(size_x=250, size_y=150)
    x_true, y_true = 120.4, 92.2
    image = get_aerial_image(
        database=database,
        x=x_true,
        y=y_true,
        z=100,
        size_x=640,
        size_y=480,
        fov_x_deg=30,
        attitude_error_std_rad=0,
        pixel_error_std=0,
    )
    result = find_position(database, image)
    print(f"Image objects: {len(image.objects)}")
    print(f"Valid: {result['valid']}")
    if not result["valid"]:
        raise RuntimeError("The example did not produce a valid position")
    error = math.hypot(result["x"] - x_true, result["y"] - y_true)
    print(f"Estimated position (m): ({result['x']:.3f}, {result['y']:.3f})")
    print(f"Position error (m): {error:.6f}")


if __name__ == "__main__":
    main()
