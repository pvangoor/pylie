from .LieGroup import LieGroup
from .SE3 import SE3 as SE3
from .SO3 import SO3 as SO3
from .R3 import R3 as R3
from .MR1 import MR1 as MR1
import numpy as np


class SIM3(LieGroup):
    # SIM(3) is defined with the matrix form
    # [ sR x ]
    # [ 0  1 ]
    def __init__(self, R=SO3(), x=R3(), s=MR1()):
        self._R = R
        self._x = x
        self._s = s

    def R(self):
        return self._R

    def x(self):
        return self._x

    def s(self):
        return self._s

    def __str__(self):
        return str(self.as_matrix())

    def Adjoint(self):
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, SIM3):
            result = SIM3()
            result._R = self._R * other._R
            result._x = self._x + self._s * (self._R * other._x)
            result._s = self._s * other._s
            return result
        elif isinstance(other, np.ndarray):
            if other.shape[0] == 3:
                return self._x + self._s * (self._R * other)
            elif other.shape[0] == 4:
                return self.as_matrix() @ other
        elif isinstance(other, SE3):
            result = SIM3()
            result._R = self._R * other._R
            result._x = self._x + self._s * (self._R * other._x)
            result._s = self._s
            return result

        return NotImplemented

    def as_matrix(self):
        mat = np.eye(4)
        mat[0:3, 0:3] = self._s._scale * self._R.as_matrix()
        mat[0:3, 3:4] = self._x._trans
        return mat
    
    def to_SE3(self):
        result = SE3()
        result._R = self._R
        result._x = self._x
        return result

    def __truediv__(self, other):
        if isinstance(other, SIM3):
            return self * other.inv()
        return NotImplemented

    def inv(self):
        result = SIM3()
        result._R = self._R.inv()
        result._x = - (self._s.inv() * (self._R.inv() * self._x))
        result._s = self._s.inv()
        return result

    def log(self):
        return NotImplemented

    @staticmethod
    def exp(sim3vec):
        assert sim3vec.shape == (7, 1), "Invalid shape of Lie algebra vector."
        return NotImplemented

    @staticmethod
    def identity() -> 'SIM3':
        result = SIM3()
        result._R = SO3.identity()
        result._x = R3.identity()
        result._s = MR1.identity()
        return result

    @staticmethod
    def valid_list_formats():
        # Possible formats are
        # SO(3) format specs
        # R(3) format specs
        # S(1) format specs
        result = dict()
        result.update(SO3.valid_list_formats())
        result.update(R3.valid_list_formats())
        result.update(MR1.valid_list_formats())
        return result

    @staticmethod
    def from_list(line, format_spec="sqx") -> 'SE3':
        result = SE3()
        SO3_formats = SO3.valid_list_formats()
        R3_formats = R3.valid_list_formats()
        MR1_formats = MR1.valid_list_formats()
        for fspec in format_spec:
            if fspec in SO3_formats:
                result._R = SO3.from_list(line, fspec)
            elif fspec in R3_formats:
                result._x = R3.from_list(line, fspec)
            elif fspec in MR1_formats:
                result._s = MR1.from_list(line, fspec)
            else:
                return NotImplemented
        return result

    def to_list(self, format_spec) -> list:
        result = []
        SO3_formats = SO3.valid_list_formats()
        R3_formats = R3.valid_list_formats()
        MR1_formats = MR1.valid_list_formats()
        for fspec in format_spec:
            if fspec in SO3_formats:
                result += self._R.to_list(fspec)
            elif fspec in R3_formats:
                result += self._x.to_list(fspec)
            elif fspec in MR1_formats:
                result += self._s.to_list(fspec)
            else:
                return NotImplemented
        return result

    @staticmethod
    def list_header(format_spec) -> list:
        result = []
        SO3_formats = SO3.valid_list_formats()
        R3_formats = R3.valid_list_formats()
        MR1_formats = MR1.valid_list_formats()
        for fspec in format_spec:
            if fspec in R3_formats:
                result += R3.list_header(fspec)
            elif fspec in SO3_formats:
                result += SO3.list_header(fspec)
            elif fspec in MR1_formats:
                result += MR1.list_header(fspec)
            else:
                return NotImplemented
        return result


if __name__ == "__main__":
    S = SIM3()
