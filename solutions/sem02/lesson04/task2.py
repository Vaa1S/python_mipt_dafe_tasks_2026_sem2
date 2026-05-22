import numpy as np


def get_dominant_color_info(
    image: np.ndarray[np.uint8],
    threshold: int = 5,
) -> tuple[np.uint8, float]:
    if threshold < 1:
        raise ValueError("threshold must be positive")

    total = image.size
    histogram = np.bincount(image.flatten(), minlength=256)

    cumulative = np.concatenate(([0], np.cumsum(histogram)))

    colors = np.arange(256)
    left = np.maximum(0, colors - threshold + 1)
    right = np.minimum(256, colors + threshold)

    counts = cumulative[right] - cumulative[left]
    counts[histogram == 0] = -1

    dominant_color = np.argmax(counts)
    percent = counts[dominant_color] / total * 100

    return np.uint8(dominant_color), percent
