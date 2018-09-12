from  mdp.abstract_mdp import AbstractMDP
import numpy as np

class berkeleyGridMDP(AbstractMDP):
    """
    A gridworld MDP taken from Berkeley lecture 
    (https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/mdps-exact-methods.pdf):

    symbol: x x x g
            x   x b
            x x x x
    
    state:  0 1 2 3
            4   5 6
            7 8 9 10 

    Where g is a goal terminal state with reward +1 and b is a "bad" terminal state with -1 reward.
    The actions are left, right, up, down. 80% of the time you travel in the intended direction,
    10% of the time you travel left of the intended direction, and 10% right. If the resulting movement
    direction is blocked, you stay put. (I.e. add the probability mass of the blocked action to the
    "stay put" event.)

    """

    # A class to represent an enum for actions
    class Actions:
        LEFT = 0
        RIGHT = 1
        UP = 2
        DOWN = 3
    # A map from action to row, col offset in the grid, for the intended direction of travel
    A_TO_ROW_COL = {Actions.LEFT  : np.array(( 0,-1)),
                    Actions.RIGHT : np.array(( 0, 1)),
                    Actions.UP    : np.array((-1, 0)),
                    Actions.DOWN  : np.array(( 1, 0))}
    # Matrix to represent the grid world
    WORLD = [[None, None, None, None, None, None],
             [None, 0,    1,    2,    3,    None],
             [None, 4,    None, 5,    6,    None],
             [None, 7,    8,    9,    10,   None],
             [None, None, None, None, None, None]]
    # A map from state number to the row and column in the matrix above
    S_TO_ROW_COL = {0  : np.array((1,1)),
                    1  : np.array((1,2)),
                    2  : np.array((1,3)),
                    3  : np.array((1,4)),
                    4  : np.array((2,1)),
                    5  : np.array((2,3)),
                    6  : np.array((2,4)),
                    7  : np.array((3,1)),
                    8  : np.array((3,2)),
                    9  : np.array((3,3)),
                    10 : np.array((3,4))}

    # The Terminal States
    TERMINAL_STATES = {3, 6}


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
        In this MDP: 11 grid spots
        """
        return 11
    
    @classmethod
    def R(cls, s, a):
        """
        The reward function.
        In this MDP, 1.0 for state g and -1 for b. 
        Args:
            s (int): The state
            a (int): The action
        Returns:
            (float): The reward for the state and action
        """
        reward = 0.0
        if s == 3:
            reward = 1.0
        if s == 6:
            reward = -1.0
        return reward

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
        # The probabilities we will return
        probabilities = np.zeros((cls.num_states()))

        # If s is terminal, return all 0.0
        if s in cls.TERMINAL_STATES:
            return probabilities

        # Get the state as an array of [row, col]
        state_r_c = cls.S_TO_ROW_COL[s]
        # Get intended next state as array of [row, col]
        intended_state_r_c = state_r_c + cls.A_TO_ROW_COL[a]
        # Get unintended next states as [array[row, col], array[row, col]]
        vertical_actions = [cls.Actions.UP, cls.Actions.DOWN]
        vertical_offsets = list(map(lambda a: cls.A_TO_ROW_COL[a], vertical_actions))
        horizontal_actions = [cls.Actions.LEFT, cls.Actions.RIGHT]
        horizontal_offsets = list(map(lambda a: cls.A_TO_ROW_COL[a], horizontal_actions))
        unintended_r_c_offsets = horizontal_offsets if a in vertical_actions else vertical_offsets
        unintended_state_r_c_list = list(map(lambda offset: state_r_c + offset, unintended_r_c_offsets))

        # Fill in the probabilities
        # Add on the probability for the intended state to the intended state 
        # (or to the current state if the intended state is None)
        r, c = intended_state_r_c
        intended_state_or_none = cls.WORLD[r][c]
        if intended_state_or_none is None:
            probabilities[s] += 0.8
        else:
            probabilities[intended_state_or_none] += 0.8
        # Add on the probability for the unintended states to the unintended state s
        # (or to the current state if the unintended state is None)
        for unintended_state_r_c in unintended_state_r_c_list:
            r, c = unintended_state_r_c
            unintended_state_or_none = cls.WORLD[r][c]
            if unintended_state_or_none is None:
                probabilities[s] += 0.1
            else:
                probabilities[unintended_state_or_none] += 0.1

        return probabilities




