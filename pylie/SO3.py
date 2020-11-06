import LieGroup
from R3 import R3 as R3
from scipy.spatial.transform import Rotation
import numpy as np

class SO3(LieGroup.LieGroup):
    def __init__(self, R = Rotation.identity()):
        self._rot = R
    
    def R(self) -> np.ndarray:
        return self._rot.as_matrix()

    def q(self) -> np.ndarray:
        return self._rot.as_quat()

    def __str__(self):
        return str(self._rot.as_matrix())
    
    def Adjoint(self):
        return self._rot.as_matrix()
    
    def __mul__(self, other):
        if isinstance(other, SO3):
            result = SO3()
            result._rot = self._rot * other._rot
            return result
        elif isinstance(other, np.ndarray) and other.shape[0] == 3:
            return self._rot.as_matrix() @ other
        elif isinstance(other, R3):
            result = R3()
            result._trans = self._rot.as_matrix() @ other._trans
            return result
        return NotImplemented
    
    def __truediv__(self, other):
        if isinstance(other, SO3):
            result = SO3()
            result._rot = self._rot * other._rot.inv()
            return result
        return NotImplemented
    
    def inv(self):
        result = SO3()
        result._rot = self._rot.inv()
        return result
    
    def log(self):
        return np.reshape(self._rot.as_rotvec(), (3,1))
    
    def as_matrix(self):
        return self._rot.as_matrix()

    @staticmethod
    def from_matrix(mat : np.ndarray) -> 'SO3':
        if not isinstance(mat, np.ndarray):
            raise TypeError
        if not mat.shape == (3,3):
            raise ValueError
        
        result = SO3()
        result._rot = Rotation.from_matrix(mat)
        return result
    
    @staticmethod
    def identity():
        result = SO3()
        result._rot = Rotation.identity()
        return result

    @staticmethod
    def exp(so3vec):
        assert so3vec.shape == (3,1), "Invalid shape of Lie algebra vector."
        result = SO3()
        result._rot = Rotation.from_rotvec(so3vec.ravel())
        return result

    @staticmethod
    def valid_list_formats() -> dict:
        # Possible formats are
        # R : 9 entry matrix (row-by-row)
        # q : 4 entry quaternion (scalar last)
        # w : 4 entry quaternion (scalar first)
        # r : 3 entry log vector
        return {'R':9, 'q':4, 'w':4, 'r':3}

    @staticmethod
    def from_list(line, format_spec="q") -> 'SO3':
        result = SO3()
        if format_spec == "R":
            mat = np.reshape(np.array([float(line[i]) for i in range(9)]), (3,3))
            result._rot = Rotation.from_matrix(mat)
            line = line[9:]
        elif format_spec == "q":
            quat = np.array([float(line[i]) for i in range(4)])
            result._rot = Rotation.from_quat(quat)
            line = line[4:]
        elif format_spec == "w":
            quat = np.array([float(line[i]) for i in [1,2,3,0]])
            result._rot = Rotation.from_quat(quat)
            line = line[4:]
        elif format_spec == "r":
            rotvec = np.array([float(line[i]) for i in range(3)])
            result._rot = Rotation.from_rotvec(rotvec)
            line = line[3:]
        else:
            return NotImplemented
        return result

    def to_list(self, format_spec) -> list:
        if format_spec == "R":
            mat = self._rot.as_matrix()
            result = mat.ravel().tolist()
        elif format_spec == "q":
            quat = self._rot.as_quat()
            result = quat.ravel().tolist()
        elif format_spec == "w":
            quat = self._rot.as_quat()
            temp = quat.ravel().tolist()
            result = [temp[i] for i in [3,0,1,2]]
        elif format_spec == "r":
            rotvec = self._rot.as_rotvec()
            result = rotvec.ravel().tolist()
        else:
            return NotImplemented
        return result
    
    @staticmethod
    def list_header(format_spec):
        if format_spec == "R":
            result = "R11,R12,R13,R21,R22,R23,R31,R32,R33".split()
        elif format_spec == "q":
            result = "qx,qy,qz,qw".split()
        elif format_spec == "w":
            result = "qw,qx,qy,qz".split()
        elif format_spec == "r":
            result = "rx,ry,rz".split()
        else:
            return NotImplemented
        return result

    @staticmethod
    def vee(mat : np.ndarray) -> np.ndarray:
        if not isinstance(mat, np.ndarray):
            raise TypeError
        if not mat.shape == (3,3):
            raise ValueError
        vec = np.array([[mat[2,1]],
                        [mat[0,2]],
                        [mat[1,0]]])
        return vec
    
    @staticmethod
    def vex(mat : np.ndarray) -> np.ndarray:
        return SO3.vee(mat)

    @staticmethod
    def wedge(vec : np.ndarray) -> np.ndarray:
        if not isinstance(vec, np.ndarray):
            raise TypeError
        if not vec.shape == (3,1):
            raise ValueError
        mat = np.array([[      0.0, -vec[2,0],  vec[1,0]],
                        [ vec[2,0],       0.0, -vec[0,0]],
                        [-vec[1,0],  vec[0,0],       0.0]])
        return mat
    
    @staticmethod
    def skew(vec : np.ndarray) -> np.ndarray:
        return SO3.wedge(vec)

if __name__ == "__main__":
    R = SO3()