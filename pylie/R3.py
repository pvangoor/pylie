from numpy.lib.arraysetops import isin
from .LieGroup import LieGroup
import numpy as np

class R3(LieGroup):
    # The Lie group of dim 3 translation.
    # [ I_3  x ]
    # [  0   1 ]
    def __init__(self, x = None):
        if x is None:
            self._trans = np.zeros((3,1))
        elif isinstance(x, R3):
            self._trans = x._trans
        elif isinstance(x, np.ndarray) or isinstance(x ,list):
            self._trans = np.reshape(x, (3,1))
        else:
            self._trans = np.zeros((3,1))
    
    def x(self) -> np.ndarray:
        return self._trans
    
    def __str__(self):
        return str(self._trans.ravel())
    
    def Adjoint(self):
        return np.eye(3)
    
    def __mul__(self, other):
        if isinstance(other, R3):
            result = R3()
            result._trans = self._trans + other._trans
            return result
        elif isinstance(other, np.ndarray):
            if other.shape[0] == 3:
                return other + self._trans
            elif other.shape[0] == 4:
                return self.as_matrix() @ other
        elif isinstance(other, float):
            return R3(self._trans * other)
        return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __add__(self, other):
        return self * other
    
    def __neg__(self):
        return self.inv()
    
    def __truediv__(self, other):
        if isinstance(other, R3):
            result = R3()
            result._trans = self._trans - other._trans
            return result
        return NotImplemented
    
    def __sub__(self, other):
        return self / other

    def inv(self):
        result = R3()
        result._trans = -self._trans
        return result
    
    def log(self):
        return self._trans
    
    def as_matrix(self):
        result = np.eye(4)
        result[0:3,3:4] = self._trans
        return result
    
    def as_vector(self):
        return self._trans.copy()
    
    @staticmethod
    def identity():
        result = R3()
        result._trans = np.zeros((3,1))
        return result

    @staticmethod
    def exp(tr3vec):
        assert tr3vec.shape == (3,1), "Invalid shape of Lie algebra vector."
        result = R3()
        result._trans = tr3vec
        return result

    @staticmethod
    def valid_list_formats() -> dict:
        # Possible formats are
        # x : 3 entry vector
        return {'x':3}

    @staticmethod
    def from_list(line, format_spec="x") -> 'R3':
        result = R3()
        if format_spec == "x":
            result._trans = np.reshape(np.array([float(line[i]) for i in range(3)]), (3,1))
            line = line[3:]
        else:
            return NotImplemented
        return result

    def to_list(self, format_spec) -> list:
        if format_spec == "x":
            result = self._trans.ravel().tolist()
        else:
            return NotImplemented
        return result
    
    @staticmethod
    def list_header(format_spec):
        if format_spec == "x":
            result = "x1,x2,x3".split()
        else:
            return NotImplemented
        return result

if __name__ == "__main__":
    x = R3()