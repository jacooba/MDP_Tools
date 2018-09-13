import numpy as np
from abc import ABC, abstractmethod
 
class AbstractSolver(ABC):
    """
    An abstract class for an MDP solver.
    """
    
    @abstractmethod
    def solve(self, mdp):
        """
        Solves the given mdp
        Returns:
            (np.array<float>): The Q values represented as a
                               2D array of size (|S|,|A|)
        """
        pass


