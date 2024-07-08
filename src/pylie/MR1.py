
from .LieGroup import LieGroup
import numpy as np
from .R3 import R3 as R3

class MR1(LieGroup):
    # The Lie group of dim 1 scaling.
    # [ s ]
    def __init__(self, s = 1.0):
        self._scale = s
    
    def scale(self):
        return self._scale
    
    def Adjoint(self):
        return np.eye(1)
    
    @staticmethod
    def adjoint(mr1vec):
        return np.array([[0]])
    
    def __mul__(self, other):
        if isinstance(other, MR1):
            result = MR1()
            result._scale = self._scale * other._scale
            return result
        elif isinstance(other, np.ndarray):
            return float(self._scale) * other
        elif isinstance(other, float):
            return float(self._scale) * other
        elif isinstance(other, R3):
            result = R3()
            result._trans = self._scale * other._trans
            return result
        return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, MR1):
            result = MR1()
            result._scale = self._scale / other._scale
            return result
        elif isinstance(other, float):
            result = self / MR1(other)
            return result
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, MR1):
            result = MR1()
            result._scale = other._scale / self._scale
            return result
        elif isinstance(other, np.ndarray):
            return other / self.as_float()
        elif isinstance(other, float):
            result = MR1(other) / self
            return result
        return NotImplemented
    
    def inv(self):
        result = MR1()
        result._scale = 1.0/self._scale
        return result
    
    def log(self):
        return np.log(self._scale)
    
    def as_matrix(self):
        return np.array(self._scale)
    
    def as_float(self):
        return float(self._scale)
    
    def __float__(self):
        return float(self._scale)
    
    @staticmethod
    def identity():
        result = MR1()
        result._scale = 1.0
        return result

    @staticmethod
    def exp(mr1vec):
        if isinstance(mr1vec, np.ndarray):
            assert mr1vec.size == 1, "The Lie algebra element should hold only 1 element."
            mr1vec = float(mr1vec)
        result = MR1()
        result._scale = np.exp(mr1vec)
        return result

    @staticmethod
    def valid_list_formats() -> dict:
        # Possible formats are
        # s : 1 entry scale
        return {'s':1}

    @staticmethod
    def from_list(line, format_spec="s") -> 'MR1':
        result = MR1()
        if format_spec == "s":
            result._scale = float(line[0])
        else:
            return NotImplemented
        return result

    def to_list(self, format_spec) -> list:
        if format_spec == "s":
            result = [float(self._scale)]
        else:
            return NotImplemented
        return result
    
    @staticmethod
    def list_header(format_spec):
        if format_spec == "s":
            result = ["s"]
        else:
            return NotImplemented
        return result

if __name__ == "__main__":
    s = MR1()
