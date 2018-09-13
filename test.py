from solver.exact_solvers import VI
from mdp.examples.acyclic_gridworld import AcyclicGridMDP
from mdp.examples.berkeley_gridworld import BerkeleyGridMDP
from mdp.examples.random_large import LargeMDP

from time import time

import unittest
import numpy as np

class MDPTests(unittest.TestCase):

    def test_VI_acyclic(self):
        # Test on acyclic MDP #
        
        # Test with both lazy and non-lazy expansion
        for is_lazy in [True, False]:
            # Test with different number of processes
            for num_process in [1, 2, 3]:
                # Test with various gamma's
                for gamma in [0.0, 0.01, 0.234, 0.50, 0.911, 0.99, 1.0]:

                    mdp = AcyclicGridMDP(gamma=gamma)
                    VI_solution = VI(mdp, H=10, lazy_expansion=is_lazy, num_process=1)
                    self.assertTrue(np.allclose(
                                        VI_solution, 
                                        mdp.calculate_known_solution()))

        # check that VI fails when number of threads is not positive
        with self.assertRaises(AssertionError):
            VI(mdp, num_process=0)
            VI(mdp, num_process=-1)

    def test_VI_Berkeley(self):
        # Test on Berkeley MDP #

        # Test with both lazy and non-lazy expansion
        for is_lazy in [True, False]:
            # Test with different number of processes
            for num_process in [1, 2, 3]:

                # Test to make sure V is the same with with various Horizons
                mdp = BerkeleyGridMDP()
                for H in sorted(BerkeleyGridMDP.HORIZON_TO_KNOWN_V.keys()):
                    VI_Q = VI(mdp, H=H, lazy_expansion=is_lazy, num_process=1)
                    VI_V = np.round(np.amax(VI_Q, axis=1), decimals=2)
                    known_V = BerkeleyGridMDP.HORIZON_TO_KNOWN_V[H]
                    self.assertEqual(list(VI_V), known_V)

                # Test final policy
                VI_final_policy = list(np.argmax(VI_Q, axis=1))
                known_final_policy = BerkeleyGridMDP.KNOWN_POLICY[:] # Make a copy
                # Replace None in known final policy with actions from the VI policy, since any are okay
                known_final_policy[3], known_final_policy[6] = VI_final_policy[3], VI_final_policy[6]
                self.assertEqual(VI_final_policy, known_final_policy)

    def test_VI_large(self):
        # Make sure multiprocessing acts reasonably on large MDP
        # (use lazy expansion since we will likely run out of RAM if not)
        
        np.random.seed(0)
        mdp = LargeMDP()

        t = time()
        Q_multi_proc = VI(mdp, H=2, lazy_expansion=True, num_process=15)
        multi_process_time = time()-t

        t = time()
        Q_single_proc = VI(mdp, H=2, lazy_expansion=True, num_process=1)
        single_process_time = time()-t

        if single_process_time < multi_process_time:
            print("WARNING: mutlti-process time (%2f) \
                   is slower on this computer than single-process time (%2f)" 
                  % (multi_process_time, single_process_time)) 

        self.assertTrue(np.allclose(Q_multi_proc, Q_single_proc))

if __name__ == '__main__':
    unittest.main()
