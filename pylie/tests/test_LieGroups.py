from ..SO3 import SO3 as SO3
from ..SE3 import SE3 as SE3
from ..SOT3 import SOT3 as SOT3
from ..SE23 import SE23 as SE23
from ..Quaternion import Quaternion as Quaternion

import numpy as np
from scipy.linalg import expm, logm
import unittest

RND_REPS = 100

class Testself(unittest.TestCase):
    Grp = SO3

    def rnd_X(self,n=1):
        if n > 1:
            return [self.Grp.exp(np.random.randn(self.Grp.DIM)) for _ in range(n)]
        else:
            return self.Grp.exp(np.random.randn(self.Grp.DIM))
    
    def rnd_v(self,n=1):
        if n > 1:
            return [np.random.randn(self.Grp.DIM) for _ in range(n)]
        else:
            return np.random.randn(self.Grp.DIM)


    def test_wedge_vee(self):
        for _ in range(RND_REPS):
            v = self.rnd_v()
            np.testing.assert_equal(self.Grp.vee(self.Grp.wedge(v)), v)

    def test_identity(self):
        for _ in range(RND_REPS):
            X = self.rnd_X()
            I = self.Grp.identity()
            np.testing.assert_almost_equal(X.as_matrix(), (X*I).as_matrix())
            np.testing.assert_almost_equal(X.as_matrix(), (I*X).as_matrix())
            np.testing.assert_almost_equal(I.as_matrix(), np.eye(*I.as_matrix().shape))

    def test_associativity(self):
        for _ in range(RND_REPS):
            X1,X2,X3 = self.rnd_X(3)
            Y1 = (X1*X2)*X3
            Y2 = X1*(X2*X3)
            np.testing.assert_almost_equal(Y1.as_matrix(), Y2.as_matrix())

    def test_matrix_product(self):
        for _ in range(RND_REPS):
            X1,X2 = self.rnd_X(2)
            Y1 = (X1*X2).as_matrix()
            Y2 = X1.as_matrix() @ X2.as_matrix()
            np.testing.assert_almost_equal(Y1, Y2)

    def test_matrix_inverse(self):
        for _ in range(RND_REPS):
            X = self.rnd_X()
            Y1 = X.inv().as_matrix()
            Y2 = np.linalg.inv(X.as_matrix())
            np.testing.assert_almost_equal(Y1, Y2)

    def test_Adjoint(self):
        for _ in range(RND_REPS):
            X = self.rnd_X()
            v = self.rnd_v()
            u1 = X.Adjoint() @ v
            u2 = self.Grp.vee(X.as_matrix() @ self.Grp.wedge(v) @ X.inv().as_matrix())
            np.testing.assert_almost_equal(u1, u2)

            Y1 = X * self.Grp.exp(v) * X.inv()
            Y2 = self.Grp.exp(u1)
            np.testing.assert_almost_equal(Y1.as_matrix(), Y2.as_matrix())

    
    def test_inverse(self):
        for _ in range(RND_REPS):
            X = self.rnd_X()
            Y1 = X * X.inv()
            Y2 = X.inv()*X
            np.testing.assert_almost_equal(Y1.as_matrix(), self.Grp.identity().as_matrix())
            np.testing.assert_almost_equal(Y2.as_matrix(), self.Grp.identity().as_matrix())

    def test_exp(self):
        for _ in range(RND_REPS):
            v = self.rnd_v()
            X1 = self.Grp.exp(v)
            X2 = self.Grp.exp(v*0.1)**10
            np.testing.assert_almost_equal(X1.as_matrix(), X2.as_matrix())

    def test_matrix_exp(self):
        for _ in range(RND_REPS):
            v = self.rnd_v()
            X1 = self.Grp.exp(v).as_matrix()
            X2 = expm(self.Grp.wedge(v))
            np.testing.assert_almost_equal(X1, X2)

    def test_log(self):
        for _ in range(RND_REPS):
            v1 = 0.01*self.rnd_v()
            v2 = self.Grp.exp(v1).log()
            np.testing.assert_almost_equal(v1, v2)

    def test_matrix_log(self):
        for _ in range(RND_REPS):
            v1 = 0.01*self.rnd_v()
            X1 = self.Grp.exp(v1).as_matrix()
            v2 = self.Grp.vee(logm(X1))

            np.testing.assert_almost_equal(v1, v2)
            
    
class TestSE3(Testself):
    Grp = SE3

class TestSOT3(Testself):
    Grp = SOT3

class TestSE23(Testself):
    Grp = SE23

class TestQuaternion(Testself):
    Grp = Quaternion

if __name__ == '__main__':
    unittest.main()