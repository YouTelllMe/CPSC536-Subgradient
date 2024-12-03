import numpy as np
from methods.subgradient import SubgradientDescent, Step

problem = SubgradientDescent((0, 0), Step.CONSTANT_SIZE, lambda x: np.sqrt(x[0]**2+x[1]**2), 10)
problem.run((-1, -1), 1)
