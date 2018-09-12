import numpy as np
import mdp.abstract_mdp

def VI(mdp, H=10, lazy_expansion=True, multi_process=False):
    """
    Solves for the Q values of the MDP using Value Iteration.
    Args:
        mdp (abstract_mdp): The mdp to solve
        H (int): The horizon, i.e. number of iterations
    Returns:
        (np.array<float>): The Q values represented as a
                           2D array of size (|S|,|A|)

    Time Complexity:    O( |S|*O(mdp.T) + H|A||S|*O(mdp.T) ) if lazy_expansion
                        O( |S|*O(mdp.T) + H|A||S|^2) )       otherwise
    Space Complexity:   O( |A||S|^2 + O(mdp.T) )             if lazy_expansion
                        O( |A||S|   + O(mdp.T) )             otherwise

    Note that if mdp.T() is implemented as just returning a row from a precomputed
    transition matrix, then the space and time complexity of mdp.T = O(|S|), and there is
    no reason NOT to use lazy_expansion.

    However, if the the space complexity of mdp.T() is O(|A||S|) and the time complexity is greater,
    as is common for a complicated transition function, then our time complexity increases by a 
    factor of O(mdp.T)/|S|, but our space complexity decreases by a factor of |S|, which can be
    critical if the state space is so large that the full transition matrix cannot be fit into memory.

    (In the analysis above, mdp.R() is considered to be constant time, since it is generally less
     computationally intensive, but the runtime could be expressed in terms of mdp.R() as well.)
    """

    # Initialize Q values
    current_Q = np.zeros((mdp.num_states(), mdp.num_actions()))
    # Initialize V values from Q
    current_V = np.amax(current_Q, axis=1)

    # Define the Transition function
    if lazy_expansion:
        T_func = mdp.T
        R_func = mdp.R
    else:
        T_matrix = mdp.calculate_T_matrix()
        T_func = lambda s, a: T_matrix[s,a]
        R_matrix = mdp.calculate_R_matrix()
        R_func = lambda s, a: R_matrix[s, a]

    def bellman_update(s, a):
        """
        Performs a bellman update in place, into current_Q using current_V.
        Args:
            s (int): The state
            a (int): The action
        """
        current_Q[s,a] = R_func(s,a) + mdp.gamma*np.dot(T_func(s,a), current_V)

    # Run VI for the given number of iterations
    for iteration in range(H):
        # Update V based on current Q
        current_V[:] = np.amax(current_Q, axis=1)
        # Update Q with the bellman update on every state and action
        for s in range(mdp.num_states()):
            for a in range(mdp.num_actions()):
                bellman_update(s,a)

    return current_Q