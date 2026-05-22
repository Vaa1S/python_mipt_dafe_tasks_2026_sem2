import numpy as np


def pad_image(image: np.ndarray, pad_size: int) -> np.ndarray:
    if pad_size < 1:
        raise ValueError

    height, width = image.shape[:2]
    new_shape = (height + 2 * pad_size, width + 2 * pad_size) + image.shape[2:]

    padded = np.zeros(new_shape, dtype=image.dtype)
    padded[pad_size:pad_size + height, pad_size:pad_size + width] = image

    return padded


def blur_image(
    image: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError

    if kernel_size == 1:
        return image.copy()

    pad_size = kernel_size // 2
    padded = pad_image(image, pad_size)

    height, width = image.shape[:2]
    accumulator = np.zeros(image.shape, dtype=np.float64)

    for di in range(kernel_size):
        for dj in range(kernel_size):
            accumulator += padded[di:di + height, dj:dj + width]

    result = accumulator / (kernel_size ** 2)

    return result.astype(image.dtype)


if __name__ == "__main__":
    import os
    from pathlib import Path

    from utils.utils import compare_images, get_image

    current_directory = Path(__file__).resolve().parent
    image = get_image(os.path.join(current_directory, "images", "circle.jpg"))
    image_blured = blur_image(image, kernel_size=21)

    compare_images(image, image_blured)