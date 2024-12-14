import numpy as np

def random_matrix(n, m):
    """
    Generates a random n by m matrix
    """
    return np.random.rand(n, m)-np.random.rand(n, m)

def random_vector(n):
    """
    Generates a random vector
    """
    return np.random.rand(n)-np.random.rand(n)


def proj_pos(s):
    """
    Given a scalar s, project onto R^+
    """
    if s> 0:
        return s
    return 0

def proj_pos_orthant(v):
    """
    Given a vector v, project onto positive orthant
    """
    return np.vectorize(proj_pos)(v)



