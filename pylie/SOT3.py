from .LieGroup import LieGroup
from .SO3 import SO3 as SO3
from .MR1 import MR1 as MR1
import numpy as np

class SOT3(LieGroup):
    def __init__(self, R : SO3 = None, s : MR1 = None):
        if R is None:
            R = SO3()
        if s is None:
            s = MR1()
        self._R = SO3(R)
        self._s = MR1(s)
    
    def R(self) -> np.ndarray:
        return self._R
    
    def s(self) -> np.ndarray:
        return self._s

    def __str__(self):
        return str(self.as_matrix())
    
    def Adjoint(self):
        return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, SOT3):
            result = SOT3()
            result._R = self._R * other._R
            result._s = self._s * other._s
            return result
        if isinstance(other, np.ndarray):
            if other.shape[0] == 3:
                return self._s * (self._R * other)
            elif other.shape[0] == 4:
                return self.as_matrix() @ other
        
        return NotImplemented
    
    @staticmethod
    def identity():
        result = SOT3()
        result._R = SO3.identity()
        result._s = MR1.identity()
        return result
    
    def as_matrix(self):
        mat = self._s.as_float() * self._R.as_matrix()
        return mat
    
    def as_quaternion(self) -> np.ndarray:
        return self.R().as_quaternion() * self._s

    @staticmethod
    def from_matrix(mat : np.ndarray) -> 'SOT3':
        if not isinstance(mat, np.ndarray):
            raise TypeError
        if not (mat.shape == (4,4) or mat.shape == (3,3)):
            raise ValueError
        
        result = SOT3()
        m = mat[0:3,0:3]
        Q = m.T @ m
        # Q is s^2 I_3.
        result._s = MR1(Q[0,0]**0.5)
        result._R = SO3.from_matrix(m / result._s)

        return result
    
    def __truediv__(self, other):
        if isinstance(other, SOT3):
            return self * other.inv()
        return NotImplemented
    
    def inv(self):
        result = SOT3()
        result._R = self._R.inv()
        result._s = self._s.inv()
        return result
    
    @staticmethod
    def exp(sot3arr) -> 'SOT3':
        if not isinstance(sot3arr, np.ndarray):
            raise TypeError
        if sot3arr.shape == (4,4):
            sot3arr = SOT3.vee(sot3arr)
        elif not sot3arr.shape == (4,1):
            raise ValueError

        result = SOT3(SO3.exp(sot3arr[0:3,0:1]), np.exp(sot3arr[3,0]))

        return result
    
    def log(self) -> np.ndarray:
        w = self._R.log()
        a = self._s.log()
        return np.vstack((w,a))

    @staticmethod
    def valid_list_formats() -> dict:
        # Possible formats are
        # q/w/R/r : SO(3) format specs
        # s : 1 entry scale
        # Q : 9 entry matrix (row-by-row)
        result = {'Q':9}
        result.update(SO3.valid_list_formats())
        result.update(MR1.valid_list_formats())
        return result

    @staticmethod
    def from_list(line, format_spec="qs") -> 'SOT3':
        result = SOT3()
        SO3_formats = SO3.valid_list_formats()
        MR1_formats = MR1.valid_list_formats()
        for fspec in format_spec:
            if fspec in SO3_formats:
                result._R = SO3.from_list(line, fspec)
                line = line[SO3_formats[fspec]:]
            elif fspec in MR1_formats:
                result._s = MR1.from_list(line, fspec)
                line = line[MR1_formats[fspec]:]
            elif fspec == "Q":
                mat = np.reshape(np.array([float(line[i]) for i in range(9)]), (3,3))
                result = SOT3.from_matrix(mat)
                line = line[9:]
            else:
                return NotImplemented
        return result

    def to_list(self, format_spec) -> list:
        result = []
        SO3_formats = SO3.valid_list_formats()
        MR1_formats = MR1.valid_list_formats()
        for fspec in format_spec:
            if fspec in SO3_formats:
                result += self._R.to_list(fspec)
            elif fspec in MR1_formats:
                result += self._s.to_list(fspec)
            elif fspec == "Q":
                result += self.as_matrix().ravel().tolist()
            else:
                return NotImplemented
        return result
    
    @staticmethod
    def list_header(format_spec) -> list:
        result = []
        SO3_formats = SO3.valid_list_formats()
        MR1_formats = MR1.valid_list_formats()
        for fspec in format_spec:
            if fspec == "Q":
                result += "Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33".split()
            elif fspec in MR1_formats:
                result += MR1.list_header(fspec)
            elif fspec in SO3_formats:
                result += SO3.list_header(fspec)
            else:
                return NotImplemented
        return result

    @staticmethod
    def vee(mat : np.ndarray) -> np.ndarray:
        if not isinstance(mat, np.ndarray):
            raise TypeError
        if not mat.shape == (3,3):
            raise ValueError
        vec = np.zeros((4,1))
        vec[0:3,:] = SO3.vex(mat)
        vec[3,0] = mat[0,0]
        return vec

    @staticmethod
    def wedge(vec : np.ndarray) -> np.ndarray:
        if not isinstance(vec, np.ndarray):
            raise TypeError
        if not vec.shape == (4,1):
            raise ValueError
        result = SO3.skew(vec[0:3,:]) + np.identity(3) * vec[3,0]
        return result

if __name__ == "__main__":
    P = SOT3()