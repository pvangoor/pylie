from .LieGroup import LieGroup
from .SO3 import SO3 as SO3
from .SE3 import SE3 as SE3
from .R3 import R3 as R3
import numpy as np

class SE23(LieGroup):
    def __init__(self, R : SO3 = None, x : R3 = None, w : R3 = None):
        if R is None:
            R = SO3()
        if x is None:
            x = R3()
        if w is None:
            w = R3()
        self._R = SO3(R)
        self._x = R3(x)
        self._w = R3(w)
    
    def R(self) -> SO3:
        return self._R
    
    def x(self) -> R3:
        return self._x
    
    def P(self) -> SE3:
        return SE3(self._R, self._x)

    def w(self) -> R3:
        return self._w

    def __str__(self) -> str:
        return str(self.as_matrix())
    
    def Adjoint(self) -> np.ndarray:
        Ad = np.zeros((9,9))
        R = self.R().as_matrix()
        Ad[0:3,0:3] = R
        Ad[3:6,0:3] = SO3.skew(self.x().as_vector()) @ R
        Ad[3:6,3:6] = R
        Ad[6:9,0:3] = SO3.skew(self.w().as_vector()) @ R
        Ad[6:9,6:9] = R
        return Ad
    
    @staticmethod
    def adjoint(se23vec : np.ndarray) -> np.ndarray:
        assert isinstance(se23vec, np.ndarray)
        assert se23vec.shape == (9,1)
        ad = np.zeros((9,9))
        OmegaCross = SO3.skew(se23vec[0:3,:])
        VCross = SO3.skew(se23vec[3:6,:])
        ACross = SO3.skew(se23vec[6:9,:])
        ad[0:3,0:3] = OmegaCross
        ad[3:6,0:3] = VCross
        ad[3:6,3:6] = OmegaCross
        ad[6:9,0:3] = ACross
        ad[6:9,6:9] = OmegaCross
        return ad
    
    def __mul__(self, other) -> 'SE23':
        if isinstance(other, SE23):
            result = SE23()
            result._R = self._R * other._R
            result._x = self._x + (self._R * other._x)
            result._w = self._w + (self._R * other._w)
            return result
        if isinstance(other, np.ndarray):
            if other.shape[0] == 5:
                return self.as_matrix() @ other
        
        return NotImplemented
    
    @staticmethod
    def identity() -> 'SE23':
        result = SE23()
        result._R = SO3.identity()
        result._x = R3.identity()
        result._w = R3.identity()
        return result
    
    def as_matrix(self) -> np.ndarray:
        mat = np.eye(5)
        mat[0:3,0:3] = self._R.as_matrix()
        mat[0:3,3:4] = self._x.as_vector()
        mat[0:3,4:5] = self._w.as_vector()
        return mat

    @staticmethod
    def from_matrix(mat : np.ndarray) -> 'SE23':
        if not isinstance(mat, np.ndarray):
            raise TypeError
        if not mat.shape == (5,5):
            raise ValueError
        
        result = SE23()
        result._R = SO3.from_matrix(mat[0:3,0:3])
        result._x = R3(mat[0:3,3:4])
        result._w = R3(mat[0:3,4:5])
        return result
    
    def __truediv__(self, other) -> 'SE23':
        if isinstance(other, SE23):
            return self * other.inv()
        return NotImplemented
    
    def inv(self) -> 'SE23':
        result = SE23()
        result._R = self._R.inv()
        result._x = - (self._R.inv() * self._x)
        result._w = - (self._R.inv() * self._w)
        return result
    
    @staticmethod
    def exp(se3arr) -> 'SE23':
        if not isinstance(se3arr, np.ndarray):
            raise TypeError
        if se3arr.shape == (5,5):
            se3arr = SE23.vee(se3arr)
        elif not se3arr.shape == (9,1):
            raise ValueError

        w = se3arr[0:3,0:1]
        theta = np.linalg.norm(w)

        if theta > 1e-6:
            A = np.sin(theta) / theta
            B = (1.0 - np.cos(theta)) / theta**2.0
            C = (1.0 - A) / theta**2.0
        else:
            A = 1.0
            B = 1.0 / 2.0
            C = 1.0 / 6.0
        
        wx = SO3.skew(w)
        wx2 = wx @ wx
        R = np.eye(3) + A * wx + B * wx2
        V = np.eye(3) + B * wx + C * wx2

        u1 = se3arr[3:6,0:1]
        u2 = se3arr[6:9,0:1]
        result = SE23(R, V@u1, V@u2)

        return result
    
    def log(self) -> np.ndarray:
        w = self._R.log()
        theta = np.linalg.norm(w)
        wx = SO3.skew(w)
        if theta > 1e-6:
            Vinv = np.eye(3) - 0.5 * wx + theta**(-2.0) * (1.0 - (theta*np.sin(theta))/(2*(1-np.cos(theta)))) * wx @ wx
        else:
            Vinv = np.eye(3) - 0.5*wx
        u1 = Vinv @ self._x.as_vector()
        u2 = Vinv @ self._w.as_vector()
        return np.vstack((w,u1,u2))

    @staticmethod
    def valid_list_formats() -> dict:
        # Possible formats are
        # q/w/R/r : SO(3) format specs
        # x : 3 entry translation
        # v : 3 entry velocity
        # P : 12 entry homogeneous matrix (row-by-row)
        result = {'P':12, 'v':3}
        result.update(SO3.valid_list_formats())
        result.update(R3.valid_list_formats())
        return result

    @staticmethod
    def from_list(line, format_spec="qx") -> 'SE3':
        result = SE3()
        SO3_formats = SO3.valid_list_formats()
        R3_formats = R3.valid_list_formats()
        for fspec in format_spec:
            if fspec in SO3_formats:
                result._R = SO3.from_list(line, fspec)
                line = line[SO3_formats[fspec]:]
            elif fspec in R3_formats:
                result._x = R3.from_list(line, fspec)
                line = line[R3_formats[fspec]:]
            elif fspec == "P":
                mat = np.reshape(np.array([float(line[i]) for i in range(12)]), (3,4))
                result._R._rot = result._R._rot.from_matrix(mat[0:3,0:3])
                result._x._trans = mat[0:3,3:4]
                line = line[12:]
            elif fspec == "v":
                result._w = R3.from_list(line, 'x')
                line = line[R3_formats['x']:]
            else:
                return NotImplemented
        return result

    def to_list(self, format_spec) -> list:
        result = []
        SO3_formats = SO3.valid_list_formats()
        R3_formats = R3.valid_list_formats()
        for fspec in format_spec:
            if fspec in SO3_formats:
                result += self._R.to_list(fspec)
            elif fspec in R3_formats:
                result += self._x.to_list(fspec)
            elif fspec == "P":
                posemat = np.hstack((self.R().as_matrix(), self.x().as_vector()))
                result += posemat.ravel().tolist()
            elif fspec == "v":
                result += self._w.to_list('x')
            else:
                return NotImplemented
        return result
    
    @staticmethod
    def list_header(format_spec) -> list:
        result = []
        SO3_formats = SO3.valid_list_formats()
        R3_formats = R3.valid_list_formats()
        for fspec in format_spec:
            if fspec == "P":
                result += "P11,P12,P13,P14,P21,P22,P23,P24,P31,P32,P33,P34".split()
            elif fspec in R3_formats:
                result += R3.list_header(fspec)
            elif fspec in SO3_formats:
                result += SO3.list_header(fspec)
            elif fspec == 'v':
                result += "v1,v2,v3".split()
            else:
                return NotImplemented
        return result

    @staticmethod
    def vee(mat : np.ndarray) -> np.ndarray:
        if not isinstance(mat, np.ndarray):
            raise TypeError
        if not mat.shape == (5,5):
            raise ValueError
        vecOmega = SO3.vex(mat[0:3,0:3])
        vecV = mat[0:3,3:4]
        vecA = mat[0:3,4:5]
        vec = np.vstack((vecOmega, vecV, vecA))
        return vec

    @staticmethod
    def wedge(vec : np.ndarray) -> np.ndarray:
        if not isinstance(vec, np.ndarray):
            raise TypeError
        if not vec.shape == (9,1):
            raise ValueError
        mat = np.zeros((5,5))
        mat[0:3,0:3] = SO3.skew(vec[0:3,0:1])
        mat[0:3,3:4] = vec[3:6, 0:1]
        mat[0:3,4:5] = vec[6:9, 0:1]
        return mat

if __name__ == "__main__":
    P = SE23()
