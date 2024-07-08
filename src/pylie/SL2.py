from .LieGroup import LieGroup
import numpy as np

class SL2(LieGroup):
    DIM = 3
    def __init__(self, H : np.ndarray = None):
        if H is None:
            self._H = np.eye(2)
        else:
            assert H.shape == (2,2), "H must be a 2x2 matrix"
            self._H = H
    
    def H(self) -> np.ndarray:
        return self._H

    def __str__(self) -> str:
        return str(self.as_matrix())
    
    def Adjoint(self) -> np.ndarray:
        a = self._H[0,0]
        b = self._H[0,1]
        c = self._H[1,0]
        d = self._H[1,1]
        return np.array([
            [2*b*c+1, -a*c, b*d],
            [-2*a*b, a*a, -b*b],
            [2*c*d, -c*c, d*d]
        ])
    
    @staticmethod
    def adjoint(sl2vec : np.ndarray) -> np.ndarray:
        assert isinstance(sl2vec, np.ndarray)
        assert sl2vec.size == 3
        u1 = sl2vec[0]
        u2 = sl2vec[1]
        u3 = sl2vec[2]
        return np.array([
            [0, -u3, u2],
            [-2*u2, 2*u1, 0],
            [2*u3, 0, -2*u1]
        ])
    
    def __mul__(self, other) -> 'SL2':
        if isinstance(other, SL2):
            result = SL2(self._H @ other._H)
            return result
        if isinstance(other, np.ndarray):
            if other.shape[0] == 2:
                return self.H @ other
        
        return NotImplemented
    
    @staticmethod
    def identity() -> 'SL2':
        return SL2(np.eye(2))
    
    def as_matrix(self) -> np.ndarray:
        return self._H

    @staticmethod
    def from_matrix(mat : np.ndarray) -> 'SL2':
        if not isinstance(mat, np.ndarray):
            raise TypeError
        if not mat.shape == (2,2):
            raise ValueError
        
        result = SL2(mat)
        return result
    
    def __truediv__(self, other) -> 'SL2':
        if isinstance(other, SL2):
            return self * other.inv()
        return NotImplemented
    
    def inv(self) -> 'SL2':
        return SL2(np.array((
                (self._H[1,1], -self._H[0,1]),
                (-self._H[1,0], self._H[0,0])
            )))
    
    @staticmethod
    def exp(sl2arr) -> 'SL2':
        if not isinstance(sl2arr, np.ndarray):
            raise TypeError
        if sl2arr.shape == (2,2):
            U = SL2.vee(sl2arr)
        elif sl2arr.size == 3:
            U = sl2arr
        else:
            raise ValueError
        
        a = U.item(0)
        b = U.item(1)
        c = U.item(2)

        theta = a**2 + b*c
        if theta > 1e-8:
            theta_sqrt = np.sqrt(theta)
            A = np.cosh(theta_sqrt)
            B = np.sinh(theta_sqrt)/theta_sqrt
        elif theta < -1e-8:
            theta_sqrt = np.sqrt(-theta)
            A = np.cos(theta_sqrt)
            B = np.sin(theta_sqrt)/theta_sqrt
        else:
            A = 1.0
            B = 1.0
        H = A * np.eye(2) + B * SL2.wedge(U)

        return SL2(H)
    
    def log(self) -> np.ndarray:
        H = self._H
        alpha = 0.5*np.trace(H)
        if alpha > 1+1e-8:
            tmp = np.arccosh(alpha)
            A = tmp / np.sinh(tmp)
        elif alpha < 1-1e-8:
            tmp = np.arccos(alpha)
            A = tmp / np.sin(tmp)
        else:
            A = 1.

        U_wedge = A * (H - alpha*np.eye(2))
        return SL2.vee(U_wedge)
        
    @staticmethod
    def vee(mat : np.ndarray) -> np.ndarray:
        if not isinstance(mat, np.ndarray):
            raise TypeError
        if not mat.shape == (2,2):
            raise ValueError

        return np.array((mat[0,0], mat[0,1], mat[1,0]))

    @staticmethod
    def wedge(vec : np.ndarray) -> np.ndarray:
        if not isinstance(vec, np.ndarray):
            raise TypeError
        if not vec.size == 3:
            raise ValueError
        mat = np.array((
            (vec.item(0), vec.item(1)),
            (vec.item(2), -vec.item(0))
        ))
        return mat

if __name__ == "__main__":
    P = SL2()