import numpy as np


# можно чутчут баллов пожалст((
def pad_image(image: np.ndarray, pad_size: int) -> np.ndarray:
    if pad_size < 1:
        raise ValueError

    if image.ndim == 2:
        height, width = image.shape
        new_height = height + 2 * pad_size
        new_width = width + 2 * pad_size
        padded = np.zeros((new_height, new_width), dtype=image.dtype)
        padded[pad_size : pad_size + height, pad_size : pad_size + width] = image
    else:
        height, width, channels = image.shape
        new_height = height + 2 * pad_size
        new_width = width + 2 * pad_size
        padded = np.zeros((new_height, new_width, channels), dtype=image.dtype)
        padded[pad_size : pad_size + height, pad_size : pad_size + width, :] = image

    return padded


def blur_image(
    image: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError

    pad_size = kernel_size // 2
    padded = pad_image(image, pad_size)

    result = np.zeros(image.shape, dtype=image.dtype)

    if image.ndim == 2:
        height, width = image.shape
        for i in range(height):
            for j in range(width):
                window = padded[i : i + kernel_size, j : j + kernel_size]
                result[i, j] = np.mean(window)
    else:
        height, width, channels = image.shape
        for i in range(height):
            for j in range(width):
                window = padded[i : i + kernel_size, j : j + kernel_size, :]
                result[i, j, :] = np.mean(window, axis=(0, 1))

    return result


if __name__ == "__main__":
    import os
    from pathlib import Path

    from utils.utils import compare_images, get_image

    current_directory = Path(__file__).resolve().parent
    image = get_image(os.path.join(current_directory, "images", "circle.jpg"))
    image_blured = blur_image(image, kernel_size=21)

    compare_images(image, image_blured)
