from ..SO3 import SO3 as SO3
import numpy as np
import unittest

RND_REPS = 100

class TestSO3(unittest.TestCase):

    @staticmethod
    def rnd_X(n=1):
        if n > 1:
            return [SO3.exp(np.random.randn(3)) for _ in range(n)]
        else:
            return SO3.exp(np.random.randn(3))
    
    @staticmethod
    def rnd_v(n=1):
        if n > 1:
            return [np.random.randn(3) for _ in range(n)]
        else:
            return np.random.randn(3)


    def test_wedge_vee(self):
        for _ in range(RND_REPS):
            v = self.rnd_v()
            np.testing.assert_equal(SO3.vee(SO3.wedge(v)), v)

    def test_associativity(self):
        for _ in range(RND_REPS):
            X1,X2,X3 = self.rnd_X(3)
            Y1 = (X1*X2)*X3
            Y2 = X1*(X2*X3)
            np.testing.assert_almost_equal(Y1.as_matrix(), Y2.as_matrix())
    
    def test_inverse(self):
        for _ in range(RND_REPS):
            X = self.rnd_X()
            Y1 = X * X.inv()
            Y2 = X.inv()*X
            np.testing.assert_almost_equal(Y1.as_matrix(), SO3.identity().as_matrix())
            np.testing.assert_almost_equal(Y2.as_matrix(), SO3.identity().as_matrix())

    def test_exp(self):
        for _ in range(RND_REPS):
            v = self.rnd_v()
            
            X1 = SO3.exp(v)
            X2 = SO3.exp(v*0.1)**10
            np.testing.assert_almost_equal(X1.as_matrix(), X2.as_matrix())
    
