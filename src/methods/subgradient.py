import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto

class Step(Enum): 
    CONSTANT_SIZE = auto() # constant step; some α
    CONSTANT_LENGTH = auto() # constant step normalized; some α/||gk||. Then ||x_(k+1)-xk|| = α (?)
    SQAURE_SUMMABLE_NON_SUMMABLE = auto() # αk >= 0, square summable (finite), not summable (diverge / infinite). Ex: α / (b+k) where a > 0, b >= 0
    NON_SUMMABLE_DIMINISHING_SIZE = auto() # αk >= 0, limit of ak to 0, not summable (diverge / infinite). Ex: α / sqrt(k)
    NON_SUMMABLE_DIMINISHING_LENGTH = auto() # step is αk / ||gk|| where αk >= 0, limit is 0, not summable (diverge / infinite).

class SubgradientDescent:
    """
    Class for Basic Subgradient Method
    """

    def __init__(self, 
                 x0: np.ndarray,
                 stepstyle: Step,
                 f, 
                 maxit: int = 1000):
        self.xcur = np.array(x0)
        self.stepstyle = stepstyle
        self.iterations = 0
        self.f = f
        self.maxit = maxit

        self.min_obj = self.get_objective()
        self.objectives = np.array([self.min_obj])
        self.terminated = False

    def iterate(self, gk: np.ndarray, alphak: float):
        """
        gk: is a subgradient direction
        alphak: stepsize > 0
        """
        gk = np.array(gk)
        if (not self.terminated):
            self.iterations += 1
            self.xcur = self.xcur - alphak * gk
            curr_objective = self.get_objective()
            self.min_obj = min(self.min_obj, curr_objective)
            self.objectives = np.append(self.objectives, curr_objective)
            self.print_statistics()

            if self.iterations > self.maxit:
                self.terminated = True
                self.plot()

    def run(self, gk: np.ndarray, alphak: float):
        gk = np.array(gk)
        self.print_statistics()
        for _ in range(self.maxit):
            # try: 
            self.iterate(gk, alphak)
            # except Exception as error:
            #     print(error)

    def print_statistics(self):
        print(f"It {self.iterations} | Current Objective: {self.objectives[-1]}, Best Objective: {self.min_obj}")

    def plot(self):
        plt.figure()
        plt.plot(range(self.iterations), self.objectives)
        plt.savefig("plot.png")

    def get_objective(self):
        return self.f(self.xcur)
    
    def get_min_objective(self):
        return self.f(self.min_obj)

class ProjectedSubgradientDescent(SubgradientDescent):
    def __init__(self, x0, stepstyle, f):
        super().__init__(x0, stepstyle, f)

class StochasticSubgradientDescent(SubgradientDescent):
    def __init__(self, x0, stepstyle, f):
        super().__init__(x0, stepstyle, f)

