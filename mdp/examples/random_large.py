from mdp.abstract_mdp import AbstractMDP
import numpy as np

class LargeMDP(AbstractMDP):
    """
    A large MDP for testing how long it takes to solve. 
    The transition and reward matrices are filled with random "junk" values.
    """

    # MDP with many states
    NUM_ACTIONS = 10
    NUM_STATES = 10000
    NUM_TERMINALS = 4
    # Fill R and T with "junk"
    R_MATRIX = np.random.rand(NUM_STATES, NUM_ACTIONS)
    T_MATRIX = np.random.rand(NUM_STATES, NUM_ACTIONS, NUM_STATES)
    # Normalize T into a distribution
    T_MATRIX = T_MATRIX/np.sum(T_MATRIX, axis=2, keepdims=True)
    # Replace terminal distributions with 0 vector
    T_MATRIX[range(NUM_TERMINALS)] = np.zeros((NUM_ACTIONS, NUM_STATES))

    def __init__(self, gamma=0.9):
        super().__init__(gamma)

    @classmethod
    def num_actions(cls):
        """
        The number of actions (i.e. |A|)
        In this MDP: left, right, up, down
        """
        return cls.NUM_ACTIONS

    @classmethod
    def num_states(cls):
        """
        The number of states (i.e. |S|)
        In this MDP: 6 grid spots
        """
        return cls.NUM_STATES
    
    @classmethod
    def R(cls, s, a):
        """
        The reward function.
        In this MDP, 1.0 for states 5 and 0 only. 
        Args:
            s (int): The state
            a (int): The action
        Returns:
            (float): The reward for the state and action
        """
        return cls.R_MATRIX[s,a]

    @classmethod
    def T(cls, s, a):
        """
        The transition function
        Args:
            s (int): The state
            a (int): The action
        Returns:
            (np.array<float>): The distribution over s'
                               as an array of length |S|,
                               if s is non-terminal.
                               The zero-vector otherwise.
        """
        return cls.T_MATRIX[s,a]
