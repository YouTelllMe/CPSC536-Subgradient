import numpy as np
import matplotlib.pyplot as plt
from helpers import Step, proj_pos_orthant, diminishing_step, constant_length, constant_size, polyak_step, polyak_step_est
from problem import N, M, f_i, g_i, f

class Subgradient:
    """
    Class for Basic Subgradient Method. Note this implementation is solving a MAX problem, so we add the subgradient instead. 
    """

    def __init__(self, 
                 x0: np.ndarray,
                 stepstyle: Step,
                 maxit: int = 1000,
                 optimal_val: float = None, 
                 S: int = 5):
        """
        S is how many iterations of consecutive nondescent (or nonascent) direction before returning to best x.
        """
        self.x_cur = np.array(x0)
        self.stepstyle = stepstyle
        self.iterations = 0
        self.maxit = maxit
        self.optimal_val = optimal_val
        self.S = S 

        self.best_obj = f(self.x_cur)
        self.x_best = self.x_cur
        self.objectives = []


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

        self.x_cur = proj_pos_orthant(self.x_cur + self.get_step(subgradient))
        return cur_obj

    def run(self):
        obj_prev = -np.inf
        self.x_best = self.x_cur
        s_counter = 0

        for _ in range(self.maxit):
            self.iterations += 1
            cur_obj = self.iterate()
            # count if decreasing 
            if (obj_prev > cur_obj): 
                s_counter += 1
            else:
                s_counter = 0
            # if decreasing consecutively, reset
            if s_counter > self.S:
                self.x_cur = self.x_best
                s_counter = 0
            obj_prev = cur_obj

    def get_step(self, subgradient):
        """
        Note: Polyak style steps don't work for incremental methods (go in circles). The paper suggests a fix which hasn't been implemented. 
        Note: Polyak step uses best current, so make sure that's updated before this is called
        """
        if self.stepstyle == Step.SQAURE_SUMMABLE_NON_SUMMABLE:
            return (diminishing_step(self.iterations)*subgradient)
        elif self.stepstyle == Step.CONSTANT_LENGTH:
            return (constant_length(self.iterations)*subgradient)
        elif self.stepstyle == Step.CONSTANT_SIZE:
            return (constant_size(self.iterations, subgradient)*subgradient)
        elif self.stepstyle == Step.POLYAK and self.optimal_val is None: 
            return (polyak_step_est(self.iterations, f(self.x_cur), self.best_obj, subgradient)*subgradient)
        elif self.stepstyle == Step.POLYAK and self.optimal_val is not None: 
            return (polyak_step(self.iterations, self.optimal_val, self.best_obj, subgradient)*subgradient)
        else: 
            raise Exception("Step is not handled.")

    def print_statistics(self):
        print(f"It {self.iterations} | Current Objective: {self.objectives[-1]}, Best Objective: {self.best_obj}, Current x: {self.x_cur}")

    