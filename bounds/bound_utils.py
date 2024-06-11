import math
import numpy as np

from scipy.stats import norm

#Code volé à Alexandre Drouin

def log_stirlings_approximation(n):
    """
    Stirling's approximation for the logarithm of the factorial

    """
    if n == 0:
        return 0
    return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)


def log_binomial_coefficient(n, k):
    """
    Logarithm of the binomial coefficient using Stirling's approximation

    """
    return (log_stirlings_approximation(n) -
            log_stirlings_approximation(k) -
            log_stirlings_approximation(n - k))

def zeta(x):
    return (6 / (np.pi * (x + 1) )**2)