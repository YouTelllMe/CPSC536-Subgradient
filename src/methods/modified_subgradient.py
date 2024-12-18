import numpy as np
from subgradient import Subgradient
from helpers import proj_pos_orthant
from problem import N, M, f_i, g_i

class ModifiedSubgradient(Subgradient):
    """
    Class for Basic Subgradient Method
    """

    def __init__(self, x0, stepstyle, maxit = 1000, optimal_val = None, S = 5, gamma = 1.5):
        self.gamma = gamma
        self.prev_subgradient = np.zeros(N)
        super().__init__(x0, stepstyle, maxit, optimal_val, S)

    def iterate(self):
        """
        Problem Specific Iteration.
        """
        cur_obj = 0
        subgradient = np.zeros(N)
        for i in range(M):
            obj_i, min_ij = f_i(self.x_cur, i)
            subgradient += g_i(i, min_ij)
            cur_obj += obj_i

        # update and store objective
        if (cur_obj > self.best_obj):
            self.best_obj = cur_obj
            self.x_best = self.x_cur
        self.objectives.append(cur_obj)

        if np.dot(self.prev_subgradient, subgradient) < 0:
            beta = -1 * self.gamma * np.dot(self.prev_subgradient, subgradient) / np.linalg.norm(self.prev_subgradient)**2
            subgradient = subgradient + beta * self.prev_subgradient

        self.x_cur = proj_pos_orthant(self.x_cur + self.get_step(subgradient))
        self.prev_subgradient = subgradient
        return cur_obj

