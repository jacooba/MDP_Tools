from abc import ABC, abstractmethod, abstractclassmethod
import numpy as np
 
class AbstractMDP(ABC):
    """
    An abstract class for an MDP.
    """

    def __init__(self, gamma):
        """
        Args:
            gamma (float): The discount factor in [0.0,1.0]
        """
        self.gamma = gamma
    
    @abstractclassmethod
    def num_actions(cls):
        """
        Get the number of actions (i.e. |A|)
        Returns:
            (int): |A|
        """
        pass

    @abstractclassmethod
    def num_states(cls):
        """
        Get the number of states (i.e. |S|)
        Returns:
            (int): |S|
        """
        pass
    
    @abstractclassmethod
    def R(cls, s, a):
        """
        The reward function.
        (Could be a function or return an entry from a precomputed R matrix)
        Args:
            s (int): The state
            a (int): The action
        Returns:
            (float): The reward for the state and action
        """
        pass

    @abstractclassmethod
    def T(cls, s, a):
        """
        The transition function.
        (Could be a function or return a row from a precomputed T matrix)
        Args:
            s (int): The state
            a (int): The action
        Returns:
            (np.array<float>): The distribution over s'
                               as an array of length |S|,
                               if s is non-terminal.
                               The zero-vector otherwise.
        """
        pass

    @classmethod
    def calculate_R_matrix(cls):
        """
        Compute the entire reward matrix using R. This is a separate function
        so that the matrix can be expanded lazily with R(s,a) to save memory.
        Returns:
            (np.array<float>): The rewards for every state action,
                               pair as a 2D array of shape (|S|,|A|)
        """
        return cls.map_over_s_a_tuples(lambda s_a_tup: cls.R(*s_a_tup))

    @classmethod
    def calculate_T_matrix(cls):
        """
        Compute the entire transition matrix using T. This is a separate function
        so that the matrix can be expanded lazily with T(s,a) to save memory.
        Returns:
            (np.array<float>): The full 3D transition matrix of shape (|S|,|A|,|S|), 
                               where entry (s, a, s') represents the probability of 
                               transitioning from s to s' given action a.
        """
        return cls.map_over_s_a_tuples(lambda s_a_tup: cls.T(*s_a_tup))

    @classmethod
    def map_over_s_a_tuples(cls, function):
        """
        Returns a numpy array that results from mapping a function: (s,a)->iterable 
        over all (s,a) tuples
        Args:
            function (lambda<(s,a)->iterable<float>>): The function to be mapped
        Returns:
            (np.array<float>): The resulting map converted to a numpy array
        """
        s_a_tuples_matrix = [[(s,a) for a in range(cls.num_actions())] for s in range(cls.num_states())]
        new_matrix = list(map(lambda inner_list: list(map(function, inner_list)), s_a_tuples_matrix))
        return np.array(new_matrix)
