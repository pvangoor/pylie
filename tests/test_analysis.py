from src.pylie import SIM3
from src.pylie import SO3
from src.pylie import analysis

import numpy as np
import unittest
from scipy.integrate import solve_ivp

np.random.seed(0)
RND_REPS = 100

class TestAnalysis(unittest.TestCase):
    def test_umeyama(self):
        for _ in range(RND_REPS):
            true_transform = SIM3.from_list(np.random.randn(7).tolist(), "srx")

            points1 = np.random.randn(3,10)
            points2 = true_transform * points1

            est_transform = analysis.umeyama(points1, points2)
            np.testing.assert_almost_equal(est_transform.as_matrix(), true_transform.as_matrix())
            
    def test_RMMK_integrate(self):

        h = 0.01  # step size
        # define a vector field in the form $\dot{R} = R f(R)$, where $R \in SO(3)$
        f = lambda R : SO3.vex(np.diag(np.diag(R.as_matrix(), 1), 1) - np.diag(np.diag(R.as_matrix(), 1), -1))

        for _ in range(RND_REPS):
            R0 = SO3.exp(np.random.randn(3))

            fI = lambda t, r : (r.reshape((3,3)) @ SO3.wedge(f(SO3(r.reshape((3,3)))))).ravel()
            soln = solve_ivp(fI, [0.,h], R0.as_matrix().ravel())
            RI = SO3(soln.y[:,-1].reshape((3,3)))

            R3 = analysis.RKMK3_integrate(R0, f, h)
            R4 = analysis.RKMK4_integrate(R0, f, h)

            self.assertLess(np.linalg.norm((RI.inv() * R3).log()), 1e-9)
            self.assertLess(np.linalg.norm((RI.inv() * R4).log()), 1e-11)

if __name__ == '__main__':
    unittest.main()