import numpy as np
from enum import Enum, auto

# STEPS #
class Step(Enum): 
    CONSTANT_SIZE = auto() # constant step; some α
    CONSTANT_LENGTH = auto() # constant step normalized; some α/||gk||. Then ||x_(k+1)-xk|| = α (?)
    SQAURE_SUMMABLE_NON_SUMMABLE = auto() # αk >= 0, square summable (finite), not summable (diverge / infinite). Ex: α / (b+k) where a > 0, b >= 0
    NON_SUMMABLE_DIMINISHING_SIZE = auto() # αk >= 0, limit of ak to 0, not summable (diverge / infinite). Ex: α / sqrt(k)
    NON_SUMMABLE_DIMINISHING_LENGTH = auto() # step is αk / ||gk|| where αk >= 0, limit is 0, not summable (diverge / infinite).
    POLYAK = auto()

def constant_length(k):
    return 0.001

def constant_size(k, subgrad):
    return 0.01 / np.linalg.norm(subgrad)

def diminishing_step(k):
    return 0.001/(k+1)

def polyak_step_est(k, f_curr, f_best, subgrad):
    """ We want to maximize f. f_best + 0.001(k+1) simulates best objective"""
    return (f_best + 1/(k+1) - f_curr) / np.linalg.norm(subgrad)**2

def polyak_step(k, f_opt, f_best, subgrad):
    return (f_opt-f_best) / np.linalg.norm(subgrad)**2


# numpy wrapper #
def random_matrix(n, m):
    """
    Generates a random n by m matrix
    """
    # return np.random.rand(n, m)-np.random.rand(n, m)
    return np.random.randint(low=0, high=100, size=(n, m))

def random_vector(n):
    """
    Generates a random vector
    """
    # return np.random.rand(n)-np.random.rand(n)
    return np.random.rand(n)

def proj_pos_orthant(v):
    """
    Given a vector v, project onto positive orthant
    """
    for i in range(len(v)):
        if v[i] < 0: 
            v[i]=0
    return np.array(v)
