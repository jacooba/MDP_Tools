from mdp.abstract_mdp import AbstractMDP
import numpy as np

class AcyclicGridMDP(AbstractMDP):
    """
    An MDP representing the following corridor gridworld:

    symbol: T x f x x T
    state:  0 1 2 3 4 5

    Where f is a fork in the road and T are terminal states.
    
    In every state other than f, every action take you towards
    the closest T. In f, left (0) moves you left, right (1) moves
    you right, and up (3) and down (4) have an equal chance of moving
    you right or left.

    The reward for any action in a terminal state is 1.0.
    The reward for all other actions is 0.0.

    The Q values should be 1.0 for both terminal states.
    It should be gamma^d for states 1, 2, 3, where d is the 
    distance to the closest terminal state.
    It should be gamma^2 and gamma^3, for left right respectively, in f.
    For up and down in f, it should be gamma * (the average of the Q values for left and right).

    """

    # A class to represent an enum for actions
    class Actions:
        LEFT = 0
        RIGHT = 1
        UP = 2
        DOWN = 3

    # The Terminal States
    TERMINAL_STATES = {0, 5}

    def __init__(self, gamma=0.9):
        super().__init__(gamma)

    @classmethod
    def num_actions(cls):
        """
        The number of actions (i.e. |A|)
        In this MDP: left, right, up, down
        """
        return 4

    @classmethod
    def num_states(cls):
        """
        The number of states (i.e. |S|)
        In this MDP: 6 grid spots
        """
        return 6
    
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
        return 1.0 if s in cls.TERMINAL_STATES else 0.0

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
        # If terminal, return all 0's
        if s in cls.TERMINAL_STATES:
            return np.zeros((cls.num_states()))
        # At for in the road
        if s == 2:
            # Up or down action
            if a in {cls.Actions.UP, cls.Actions.DOWN}:
                return np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.0])
            # Left
            if a == cls.Actions.LEFT:
                return np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            # right
            return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        # Deterministically move towards closest (left) terminal state
        if s == 1:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Deterministically move towards closest (right) terminal state
        to_return = np.zeros((cls.num_states()));
        to_return[s+1] = 1.0
        return to_return

    def calculate_known_solution(self):
        """
        Gives the known solution to this MDP
        Returns:
            (np.array<float>): The Q values represented as a
                               2D array of size (|S|,|A|)
        """
        g = self.gamma # easier to write
        # V (except V[3] is wrong below. It is currently the Q value for up or down)
        V = np.array([1.0, g**1, 0.5*(g**2+g**3), g**2, g**1, 1.0])
        # Q values are the same as V
        Q = np.reshape(V, (self.num_states(), 1))*np.ones((self.num_states(), self.num_actions()))
        # fix Q for left and right from f
        Q[2, 0] = g**2 
        Q[2, 1] = g**3
        return Q


