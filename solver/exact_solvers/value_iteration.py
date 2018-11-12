import multiprocessing
import itertools
import numpy as np

from multiprocessing import Process, Array
from ctypes import c_double
from solver.abstract_solver import AbstractSolver

class ValueIteration(AbstractSolver):

    def __init__(self, H=10, lazy_expansion=True, num_process=1):
        """
        Store the parameters for this VI solver
        Args:
            H (int): The horizon, i.e. number of iterations
            lazy_expansion (bool): Whether to use the transition 
                                   function as necessary or cache it
            num_process (int): The number of processes to use
        """
        self.H = H
        self.lazy_expansion = lazy_expansion
        self.num_process = num_process

    def solve(self, mdp):
        """
        Solves for the Q values of the MDP using Value Iteration.
        Args:
            mdp (AbstractMDP): The mdp to solve
        Returns:
            (np.array<float>): The Q values represented as a
                               2D array of size (|S|,|A|)

        Time Complexity:    O( |S|*O(mdp.T) + H|A||S|*O(mdp.T) ) if lazy_expansion
                            O( |S|*O(mdp.T) + H|A||S|^2) )       otherwise
        Space Complexity:   O( |A||S|   + O(mdp.T) )             if lazy_expansion
                            O( |A||S|^2 + O(mdp.T) )             otherwise

        Note that if mdp.T() is implemented as just returning a row from a precomputed
        transition matrix, then the time complexity of mdp.T = O(|S|) and the space complexity
        of mdp.T = O(|A||S|^2), and there is no reason NOT to use lazy_expansion.

        However, if the the space complexity of mdp.T() is O(|A||S|) and the time complexity is greater,
        as is common for a complicated transition function, then our time complexity increases by a 
        factor of O(mdp.T)/|S|, but our space complexity decreases by a factor of |S|, which can be
        critical if the state space is so large that the full transition matrix cannot be fit into memory.

        (In the analysis above, mdp.R() is considered to be constant time, since it is generally less
         computationally intensive, but the runtime could be expressed in terms of mdp.R() as well.)
        """
        assert self.num_process > 0, "The number of processes must be > 0"

        # Initialize Q values (use shared memory if multi-processing)
        current_Q = np.zeros((mdp.num_states(), mdp.num_actions()))
        if self.num_process > 1:
            current_Q = np.frombuffer(Array(c_double, current_Q.flat, lock=False))
            current_Q.resize(mdp.num_states(), mdp.num_actions())
        # Initialize V values from Q
        current_V = np.amax(current_Q, axis=1)

        # Define the Transition function (lazy or not lazy)
        if self.lazy_expansion:
            T_func, R_func = mdp.T, mdp.R
        else:
            T_matrix, R_matrix = mdp.calculate_T_matrix(), mdp.calculate_R_matrix()
            T_func, R_func = lambda s, a: T_matrix[s,a], lambda s, a: R_matrix[s,a]

        # Get all state action (s,a) tuples
        s_a_tuples = list(itertools.product(range(mdp.num_states()), range(mdp.num_actions())))
        # Run VI on all (s,a) tuples for the given number of iterations
        for iteration in range(self.H):
            print("On iteration ", iteration, "out of ", self.H)
            # Update V based on current Q
            current_V[:] = np.amax(current_Q, axis=1)
            # Update Q with the bellman update on every state and action
            if self.num_process == 1:
                for s, a in s_a_tuples:
                    current_Q[s,a] = ValueIteration._bellman_update(s, a, R_func, T_func, current_V, mdp.gamma)
            else:
                workers = []
                for worker_num in range(self.num_process):
                    worker = Process(target=ValueIteration._bellman_worker, 
                                     args=(s_a_tuples, worker_num, self.num_process, current_Q,
                                           R_func, T_func, current_V, mdp.gamma))
                    workers.append(worker)
                    worker.start()
                for worker in workers:
                    worker.join()

        return current_Q

    @staticmethod
    def _bellman_worker(s_a_tuples, worker_num, num_process, current_Q, R_func, T_func, current_V, gamma):
        """
        Performs a given fraction of the bellman updates in place, into current_Q using current_V.
        Args:
            s_a_tuples (list<tuple<int,int>>): A list of all (s,a) pairs
            R_func (s,a->float): The reward function
            T_func (s,a->np.array<float>): The transition function
            current_V (np.array<float): The current value function
            gamma (float): The discount factor in [0.0,1.0]
        """
        start_num = int(len(s_a_tuples) * (worker_num/num_process))
        end_num = int(len(s_a_tuples) * ((worker_num+1)/num_process))
        for i in range(start_num, end_num):
            s, a = s_a_tuples[i]
            current_Q[s,a] = ValueIteration._bellman_update(s, a, R_func, T_func, current_V, gamma)

    @staticmethod
    def _bellman_update(s, a, R_func, T_func, current_V, gamma):
        """
        Returns the value of a bellman update for a state and action using a current value.
        Args:
            s (int): The state
            a (int): The action
            R_func (s,a->float): The reward function
            T_func (s,a->np.array<float>): The transition function
            current_V (np.array<float): The current value function
            gamma (float): The discount factor in [0.0,1.0]
        Return:
            (float): The new value that Q[s,a] should have after the update
        """
        return R_func(s,a) + gamma*np.dot(T_func(s,a), current_V)


