from mdp.abstract_mdp import AbstractMDP
import numpy as np

class RandomLargeMDP(AbstractMDP):
    """
    A large MDP for testing how long it takes to solve. 
    The transition and reward matrices are filled with random "junk" values.
    (Set up to be different values for every instance)
    """

    # Define how large MDP is
    _NUM_ACTIONS = 10
    _NUM_STATES = 10000
    _NUM_TERMINALS = 4

    def __init__(self, gamma=0.9):
        super().__init__(gamma)
        # Fill R and T with "junk"
        t_matrix_shape = (self.__class__._NUM_STATES, 
                          self.__class__._NUM_ACTIONS, 
                          self.__class__._NUM_STATES)
        self._r_matrix = np.random.rand(*(t_matrix_shape[:2]))
        self._t_matrix = np.random.rand(*t_matrix_shape)
        # Normalize T into a distribution
        self._t_matrix = self._t_matrix/np.sum(self._t_matrix, axis=2, keepdims=True)
        # Replace terminal distributions with 0 vector
        self._t_matrix[range(self.__class__._NUM_TERMINALS)] = np.zeros(t_matrix_shape[1:])

    def num_actions(self):
        """
        The number of actions (i.e. |A|)
        In this MDP: left, right, up, down
        """
        return self.__class__._NUM_ACTIONS

    def num_states(self):
        """
        The number of states (i.e. |S|)
        In this MDP: 6 grid spots
        """
        return self.__class__._NUM_STATES
    
    def R(self, s, a):
        """
        The reward function.
        In this MDP, 1.0 for states 5 and 0 only. 
        Args:
            s (int): The state
            a (int): The action
        Returns:
            (float): The reward for the state and action
        """
        return self._r_matrix[s,a]

    def T(self, s, a):
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
        return self._t_matrix[s,a]
