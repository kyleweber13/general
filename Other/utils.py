import numpy as np
import bottleneck


def get_zncc(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Get zero-normalized cross-correlation (ZNCC) of the given vectors.
    Adapted from: https://en.wikipedia.org/wiki/Cross-correlation#Zero-normalized_cross-correlation_(ZNCC)
    To improve performance, the formula given by Wikipedia is rearranged as follows::
        1 / (n * std(x) * std(y)) * (sum[x(i) * y(i)] - n * mean(x) * mean(y))
    """
    # Ensure x is the longer signal.
    x, y = sorted((x, y), key=len, reverse=True)

    # Calculate rolling mean and standard deviation. Discard first few NaN values.
    x_mean = bottleneck.move_mean(x, len(y))[len(y) - 1:]
    x_std = bottleneck.move_std(x, len(y))[len(y) - 1:]

    # Avoid division by zero and numerical errors caused by zero or very small standard deviations.
    x_std_reciprocal = np.reciprocal(x_std, where=np.abs(x_std) > 0.0000001)

    y_mean = np.mean(y)
    y_std_reciprocal = 1 / np.std(y)

    n = len(y)

    # Calculate correlation and normalize.
    correlation = np.correlate(x, y, mode='valid')

    z = (1 / n) * x_std_reciprocal * y_std_reciprocal * (correlation - n * x_mean * y_mean)

    return z
