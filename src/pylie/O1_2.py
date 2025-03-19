from .LieGroup import LieGroup
import numpy as np

# The indefinite orthogonal group of signature 1,2
class O1_2(LieGroup):
    DIM = 3
    _cross_mat = np.array(((0,-1),(1,0)))
    def __init__(self, L : np.ndarray = None):
        if L is None:
            self._L = np.eye(3)
        else:
            assert L.shape == (3,3), "L must be a 3x3 matrix"
            self._L = L
    
    def L(self) -> np.ndarray:
        return self._L
    
    def _extract_components(self):
        return self._L[1:,1:], self._L[0,1:], self._L[1:,0], self._L[0,0]
    
    def _validity(self):
        # Check how valid a given matrix is. Returns a nonnegative float that should be zero.
        g = np.eye(3)
        g[1:,1:] = -np.eye(2)
        error_mat = self._L.T @ g @ self._L - g
        return np.linalg.norm(error_mat)


    def __str__(self) -> str:
        return str(self.as_matrix())
    
    def Adjoint(self) -> np.ndarray:
        Ad_mat = np.empty((3,3))
        A,b,c,d = self._extract_components()
        Ad_mat[0,0] = -0.5 * np.trace(self._cross_mat @ A @ self._cross_mat @ A.T)
        Ad_mat[1:,0] = -A @ self._cross_mat @ b
        Ad_mat[0,1:] = A.T @ self._cross_mat.T @ c
        Ad_mat[1:,1:] = d * A - np.outer(c,b)
        return Ad_mat
        
    
    @staticmethod
    def adjoint(o1_2vec : np.ndarray) -> np.ndarray:
        assert isinstance(o1_2vec, np.ndarray)
        assert o1_2vec.size == 3
        ad_mat = np.empty((3,3))
        ad_mat[0,0] = 0.0
        omega = o1_2vec[0]; u = o1_2vec[1:]
        ad_mat[1:,0] = - O1_2._cross_mat @ u
        ad_mat[0,1:] = O1_2._cross_mat.T @ u
        ad_mat[1:,1:] = O1_2._cross_mat * omega
        return ad_mat
    
    def __mul__(self, other) -> 'O1_2':
        if isinstance(other, O1_2):
            result = O1_2(self._L @ other._L)
            return result
        if isinstance(other, np.ndarray):
            if other.shape[0] == 3:
                return self.L @ other
        
        return NotImplemented
    
    @staticmethod
    def identity() -> 'O1_2':
        return O1_2(np.eye(3))
    
    def as_matrix(self) -> np.ndarray:
        return self._L

    @staticmethod
    def from_matrix(mat : np.ndarray) -> 'O1_2':
        if not isinstance(mat, np.ndarray):
            raise TypeError
        if not mat.shape == (3,3):
            raise ValueError
        
        result = O1_2(mat)
        return result
    
    def __truediv__(self, other) -> 'O1_2':
        if isinstance(other, O1_2):
            return self * other.inv()
        return NotImplemented
    
    def inv(self) -> 'O1_2':
        A,b,c,d = self._extract_components()
        LInv = np.empty((3,3))
        LInv[1:,1:] = A.T
        LInv[0,0] = d
        LInv[0,1:] = -c
        LInv[1:,0] = -b
        return O1_2(LInv)
    
    @staticmethod
    def exp(O1_2vec) -> 'O1_2':
        if not isinstance(O1_2vec, np.ndarray):
            raise TypeError
        if O1_2vec.shape == (3,3):
            u = O1_2.vee(O1_2vec)
        elif O1_2vec.size == 3:
            u = O1_2vec
        else:
            raise ValueError
        
        q2 = u[1]**2 + u[2]**2 - u[0]**2

        U = O1_2.wedge(u)
        if q2 == 0.:
            L = np.eye(3) + U + 0.5 * U@U
        elif q2 > 0:
            q = np.sqrt(q2)
            L = np.eye(3) + np.sinh(q) / q * U + (np.cosh(q) - 1)/q2 * U@U
        else:
            iq = np.sqrt(-q2)
            L = np.eye(3) + np.sin(iq) / iq * U + (np.cos(iq) - 1)/q2 * U@U

        return O1_2(L)
    
    def log(self) -> np.ndarray:
        A,b,c,d = self._extract_components()
        sinh_q_2 = np.linalg.norm(0.5*(b+c))**2 - 0.25 * (A[0,1] - A[1,0])**2
        if sinh_q_2 == 0:
            term_1 = 0
        elif sinh_q_2 > 0:
            q = np.arcsinh(np.sqrt(sinh_q_2))
            term_1 = np.sinh(q) / q
        else:
            iq = np.arcsin(np.sqrt(-sinh_q_2))
            term_1 = np.sin(iq)/iq
        
        U = np.empty(3)
        U[0] = (A[1,0]-A[0,1]) / (2*term_1)
        U[1:] = (b+c) / (2*term_1)

        return U
        
    @staticmethod
    def vee(mat : np.ndarray) -> np.ndarray:
        if not isinstance(mat, np.ndarray):
            raise TypeError
        if not mat.shape == (3,3):
            raise ValueError

        return np.array((-mat[1,2], mat[0,1], mat[0,2]))

    @staticmethod
    def wedge(vec : np.ndarray) -> np.ndarray:
        if not isinstance(vec, np.ndarray):
            raise TypeError
        if not vec.size == 3:
            raise ValueError
        mat = np.array((
            (0, vec.item(1), vec.item(2)),
            (vec.item(1), 0, -vec.item(0)),
            (vec.item(2), vec.item(0), 0)
        ))
        return mat

if __name__ == "__main__":
    P = O1_2()