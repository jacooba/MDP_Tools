from solver.exact_solvers import VI
from mdp.examples.acyclic_gridworld import acyclicGridMDP
from mdp.examples.berkeley_gridworld import berkeleyGridMDP

import unittest
import numpy as np

class MDPTests(unittest.TestCase):

    def test_VI_acyclic(self):
        # Test on acyclic MDP #
        
        # Test with both lazy and non-lazy expansion
        for lazy in [True, False]:
            # Test with various gamma's
            for gamma in [0.0, 0.01, 0.234, 0.50, 0.911, 0.99, 1.0]:
                mdp = acyclicGridMDP(gamma=gamma)
                VI_solution = VI(mdp, H=10, lazy_expansion=lazy, multi_process=False)
                self.assertTrue(np.allclose(
                                    VI_solution, 
                                    mdp.calculate_known_solution()))

    def test_VI_Berkeley(self):
        # Test on Berkeley MDP #

        # Test with both lazy and non-lazy expansion
        for lazy in [True, False]:

            # Test to make sure V is the same with with various Horizons
            mdp = berkeleyGridMDP()
            horizon_to_V = {0    : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            1    : [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                            2    : [0.0, 0.0, .72, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                            3    : [0.0, 0.52, .78, 1.0, 0.0, 0.43, -1.0, 0.0, 0.0, 0.0, 0.0],
                            100  : [.64, .74, .85, 1.0, .57, .57, -1.0, .49, .43, .48, .28],
                            1000 : [.64, .74, .85, 1.0, .57, .57, -1.0, .49, .43, .48, .28]}
            for H in sorted(horizon_to_V.keys()):
                VI_Q = VI(mdp, H=H, lazy_expansion=lazy, multi_process=False)
                VI_V = np.round(np.amax(VI_Q, axis=1), decimals=2)
                known_V = horizon_to_V[H]
                self.assertEqual(list(VI_V), known_V)

            # Test final policy
            VI_final_policy = list(np.argmax(VI_Q, axis=1))
            known_final_policy = [1, 1, 1, None, 2, 2, None, 2, 0, 2, 0]
            # Replace None in known final policy with actions from the VI policy, since any are okay
            known_final_policy[3], known_final_policy[6] = VI_final_policy[3], VI_final_policy[6]
            self.assertEqual(VI_final_policy, known_final_policy)

if __name__ == '__main__':
    unittest.main()