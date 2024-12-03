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
                 f):
        self.xcur = x0
        self.stepstyle = stepstyle
        self.iterations = 0
        self.f = f
        self.objectives = np.array([])
        self.min_obj = self.get_objective()

    def iterate(self, gk: np.ndarray, alphak: float):
        """
        gk: is a subgradient direction
        alphak: stepsize > 0
        """
        self.iterations += 1
        self.xcur = self.xcur - alphak * gk
        curr_objective = self.get_objective()
        self.min_obj = min(self.min_obj, curr_objective)
        np.append(self.objectives, curr_objective)

    def get_objective(self):
        return self.f(self.xcur)

    def print_statistics(self):
        print("Hello!")

    def plot(self):
        plt.figure()
        plt.plot(range(self.iterations), self.objectives)
        plt.savefig("plot.png")

class ProjectedSubgradientDescent(SubgradientDescent):
    def __init__(self, x0, stepstyle, f):
        super().__init__(x0, stepstyle, f)

class StochasticSubgradientDescent(SubgradientDescent):
    def __init__(self, x0, stepstyle, f):
        super().__init__(x0, stepstyle, f)


