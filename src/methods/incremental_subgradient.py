import numpy as np
from subgradient import Subgradient
from helpers import proj_pos_orthant
from problem import N, M, f_i, g_i, f

class IncrementalSubgradient(Subgradient):
    """
    Class for Basic Subgradient Method
    """

    def iterate(self):
        """
        Problem Specific Incremental Iteration.
        """
        for i in range(M):
            _, min_ij = f_i(self.x_cur, i)
            subgradient = g_i(i, min_ij)
            obj = f(self.x_cur)
            if (obj > self.best_obj): 
                self.obj_best = obj
                self.x_best = self.x_cur
            self.x_cur = proj_pos_orthant(self.x_cur + self.get_step(subgradient))

        cur_obj = f(self.x_cur)
        if (cur_obj > self.best_obj):
            self.best_obj = cur_obj
            self.x_best = self.x_cur
        self.objectives.append(cur_obj)

        return cur_obj

