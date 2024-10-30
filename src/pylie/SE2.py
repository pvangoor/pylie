from .LieGroup import LieGroup
import numpy as np

class SE2(LieGroup):
    DIM = 3
    _cross_matrix = np.array(((0.,-1.),(1.,0.)))
    def __init__(self, theta : float = 0., p : np.ndarray = None):
        if p is None:
            p = np.zeros(2)
        self._theta = SE2._wrap_angle(theta)
        self._p = p
    
    def theta(self) -> float:
        return self._theta
    
    def R(self) -> np.ndarray:
        return SE2._rotation_matrix(self._theta)
    
    @staticmethod
    def _rotation_matrix(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array((
            (c, -s),
            (s, c)
        ))
    
    @staticmethod
    def _wrap_angle(t):
        return t - np.floor((t + np.pi) / (2*np.pi)) * (2*np.pi)

    
    def p(self) -> np.ndarray:
        return self._p

    def __str__(self) -> str:
        return str(self.as_matrix())
    
    def Adjoint(self) -> np.ndarray:
        Ad = np.eye(3)
        Ad[1:,0] = -self._cross_matrix @ self._p
        Ad[1:,1:] = self.R()
        return Ad
    
    @staticmethod
    def adjoint(se2vec : np.ndarray) -> np.ndarray:
        assert isinstance(se2vec, np.ndarray)
        assert se2vec.size == 3
        ad = np.zeros((3,3),dtype=se2vec.dtype)
        ad[1:,0] = -SE2._cross_matrix @ se2vec[1:]
        ad[1:,1:] = SE2._cross_matrix * se2vec[0]
        return ad
    
    def __mul__(self, other) -> 'SE2':
        if isinstance(other, SE2):
            result = SE2()
            result._theta = SE2._wrap_angle(self._theta + other._theta)
            result._p = self._p + self.R() @ other._p
            return result
        if isinstance(other, np.ndarray):
            if other.shape[0] == 2:
                return self._p + self.R() @ other
            elif other.shape[0] == 3:
                return self.as_matrix() @ other
        
        return NotImplemented
    
    @staticmethod
    def identity() -> 'SE2':
        result = SE2()
        result._theta = 0.
        result._p = np.zeros(2)
        return result
    
    def as_matrix(self) -> np.ndarray:
        mat = np.eye(3)
        mat[0:2,0:2] = self.R()
        mat[0:2,2] = self._p
        return mat

    @staticmethod
    def from_matrix(mat : np.ndarray) -> 'SE2':
        if not isinstance(mat, np.ndarray):
            raise TypeError
        if not mat.shape == (3,3):
            raise ValueError
        
        result = SE2()
        result._theta = np.arctan2(mat[1,0], mat[0,0])
        result._p = mat[0:2,2]
        return result
    
    def as_vector(self) -> np.ndarray:
        return np.array([self._theta, self._p[0], self._p[1]])
    
    @staticmethod
    def from_vector(mat : np.ndarray) -> 'SE2':
        if not isinstance(mat, np.ndarray):
            raise TypeError
        if not mat.size == 3:
            raise ValueError
        
        result = SE2()
        result._theta = mat.item(0)
        result._p = np.array([mat.item(1),mat.item(2)])
        return result

    def __truediv__(self, other) -> 'SE2':
        if isinstance(other, SE2):
            return self * other.inv()
        return NotImplemented
    
    def inv(self) -> 'SE2':
        result = SE2()
        result._theta = - self._theta
        result._p = - self.R().T @ self._p
        return result
    
    @staticmethod
    def exp(se2arr) -> 'SE2':
        if not isinstance(se2arr, np.ndarray):
            raise TypeError
        if se2arr.shape == (3,3):
            se2arr = SE2.vee(se2arr)
        elif not se2arr.size == 3:
            raise ValueError

        w = se2arr[0]

        if abs(w) > 1e-6:
            A = np.sin(w) / w
            B = (1.0 - np.cos(w)) / w**2.0
        else:
            A = 1.0
            B = 1.0 / 2.0
        
        U = SE2.wedge(se2arr)
        

        mat = np.eye(3) + A * U + B * U @ U
        result = SE2.from_matrix(mat)
        result._theta = w
        return result
    
    def log(self) -> np.ndarray:
        w = self._theta
        if abs(w) > 1e-6:
            v = w / (2*(1-np.cos(w))) * self._cross_matrix.T @ (np.eye(2) - self.R().T) @ self._p
        else:
            v = self._p
        return np.concatenate(([w], v))
        
    @staticmethod
    def vee(mat : np.ndarray) -> np.ndarray:
        if not isinstance(mat, np.ndarray):
            raise TypeError
        if not mat.shape == (3,3):
            raise ValueError
        vecOmega = -mat[0,1]
        vecV = mat[0:2,2]
        return np.concatenate(([vecOmega], vecV))

    @staticmethod
    def wedge(vec : np.ndarray) -> np.ndarray:
        if not isinstance(vec, np.ndarray):
            raise TypeError
        if not vec.size == 3:
            raise ValueError
        mat = np.zeros((3,3),dtype=vec.dtype)
        mat[0:2,0:2] = vec[0] * SE2._cross_matrix
        mat[0:2,2] = vec.ravel()[1:]
        return mat

if __name__ == "__main__":
    P = SE2()