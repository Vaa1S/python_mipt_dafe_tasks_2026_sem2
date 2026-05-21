import numpy as np


def get_dominant_color_info(
    image: np.ndarray[np.uint8],
    threshold: int = 5,
) -> tuple[np.uint8, float]:
    if threshold < 1:
        raise ValueError

    flat = image.flatten()
    total = flat.size

    counts = np.zeros(256, dtype=np.int64)
    for color in range(256):
        diff = np.abs(flat.astype(np.int64) - color)
        counts[color] = np.sum(diff < threshold)

    most_common = np.argmax(counts)
    percent = counts[most_common] / total * 100

    return np.uint8(most_common), percent
