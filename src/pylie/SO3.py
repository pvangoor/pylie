
from .LieGroup import LieGroup
from .R3 import R3 as R3
from scipy.spatial.transform import Rotation
import numpy as np

class SO3(LieGroup):
    DIM = 3
    def __init__(self, R = None):
        if R is None:
            self._R = np.eye(3)
        elif isinstance(R, np.ndarray):
            self._R = R
        elif isinstance(R, SO3):
            self._R = R._R
        elif isinstance(R, Rotation):
            self._R = R.as_matrix()
            
    @staticmethod
    def adjoint(so3vec : np.ndarray) -> np.ndarray:
        assert isinstance(so3vec, np.ndarray)
        assert so3vec.size == 3
        return SO3.skew(so3vec)
            
    def R(self) -> np.ndarray:
        return self._R

    def as_quaternion(self) -> np.ndarray:
        return Rotation.from_matrix(self._R).as_quat()

    def as_euler(self, seq='xyz', degrees=True) -> np.ndarray:
        return Rotation.from_matrix(self._R).as_euler(seq, degrees=degrees)

    @staticmethod
    def from_euler(euler_angles, seq='xyz', degrees=True) -> np.ndarray:
        return SO3.from_matrix(Rotation.from_euler(seq=seq, angles=euler_angles, degrees=degrees).as_matrix())

    def __str__(self):
        return str(self._R)
    
    def Adjoint(self):
        return self.as_matrix()
    
    def __mul__(self, other):
        if isinstance(other, SO3):
            return SO3(self._R @ other._R)
        elif isinstance(other, np.ndarray) and other.shape[0] == 3:
            return self.as_matrix() @ other
        elif isinstance(other, R3):
            result = R3()
            result._trans = self._R @ other._trans
            return result
        return NotImplemented
    
    def __truediv__(self, other):
        if isinstance(other, SO3):
            return self * other.inv()
        return NotImplemented
    
    def inv(self):
        return SO3(self._R.T)
    
    def log(self):
        t = (np.trace(self._R) - 1.0) / 2.0
        if abs(t) > 1.0:
            t = t / abs(t)
        theta = np.arccos(t)
        if abs(theta) > 1e-6:
            coefficient = theta / (2.0 * np.sin(theta))
        else:
            coefficient = 0.5

        Omega = coefficient * (self._R - self._R.transpose())
        return SO3.vee(Omega)
    
    def as_matrix(self):
        return self._R

    @staticmethod
    def from_matrix(mat : np.ndarray) -> 'SO3':
        if not isinstance(mat, np.ndarray):
            raise TypeError
        if not mat.shape == (3,3):
            raise ValueError
        
        return SO3(mat)
    
    @staticmethod
    def identity():
        return SO3(np.eye(3))

    @staticmethod
    def exp(so3vec):
        assert so3vec.size == 3, "The so(3) Lie algebra vector must have 3 elements."
        theta = np.linalg.norm(so3vec)
        if theta == 0.0:
            return SO3(np.eye(3))
        elif theta < 1e-6:
            t1 = 1.0 - theta**2/6.0
            t2 = 0.5 - theta**2/24.0
        else:
            t1 = np.sin(theta) / theta
            t2 = (1.0 - np.cos(theta)) / (theta**2)

        Omega = SO3.wedge(so3vec)
        return SO3(np.eye(3) + t1 * Omega + t2 * Omega @ Omega)

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
            result._R = mat
            line = line[9:]
        elif format_spec == "q":
            quat = np.array([float(line[i]) for i in range(4)])
            result._R = Rotation.from_quat(quat).as_matrix()
            line = line[4:]
        elif format_spec == "w":
            quat = np.array([float(line[i]) for i in [1,2,3,0]])
            result._R = Rotation.from_quat(quat).as_matrix()
            line = line[4:]
        elif format_spec == "r":
            rotvec = np.array([float(line[i]) for i in range(3)])
            result = SO3.exp(rotvec)
            line = line[3:]
        else:
            return NotImplemented
        return result

    def to_list(self, format_spec) -> list:
        if format_spec == "R":
            mat = self.as_matrix()
            result = mat.ravel().tolist()
        elif format_spec == "q":
            quat = self.as_quat()
            result = quat.ravel().tolist()
        elif format_spec == "w":
            quat = self.as_quat()
            temp = quat.ravel().tolist()
            result = [temp[i] for i in [3,0,1,2]]
        elif format_spec == "r":
            rotvec = self.log()
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
        vec = np.array([mat[2,1],
                        mat[0,2],
                        mat[1,0]])
        return vec
    
    @staticmethod
    def vex(mat : np.ndarray) -> np.ndarray:
        return SO3.vee(mat)

    @staticmethod
    def wedge(vec : np.ndarray) -> np.ndarray:
        if not isinstance(vec, np.ndarray):
            raise TypeError
        if not vec.size == 3:
            raise ValueError
        mat = np.array([[      0.0, -vec.item(2),  vec.item(1)],
                        [ vec.item(2),       0.0, -vec.item(0)],
                        [-vec.item(1),  vec.item(0),       0.0]])
        return mat
    
    @staticmethod
    def skew(vec : np.ndarray) -> np.ndarray:
        return SO3.wedge(vec)
    
    @staticmethod
    def from_vectors(origin : np.ndarray, dest : np.ndarray) -> 'SO3':
        origin = origin / np.linalg.norm(origin)
        dest = dest / np.linalg.norm(dest)
        v = np.cross(origin, dest, axis=0);
        c = origin.T @ dest
        s = np.linalg.norm(v)

        if (abs(1 + c) <= 1e-8):
            # The vectors are exactly opposing
            # We need to find a vector perpendicular to origin
            # e1 will work almost always. Otherwise e2 will.
            v2 = np.reshape((1.0,0,0), (3,1))
            if abs(float(v2.T @ origin)) > 1 - 1e-8:
                v2 = np.reshape((0,1.0,0), (3,1))
            v2 = np.cross(origin, v2, axis=0) # This is perpendicular to origin and not zero
            v2 = v2 / np.linalg.norm(v2)
            # Rotating around v2 by pi rads will give the desired result
            R = SO3.exp(np.pi * v2)

        else:
            vx = SO3.skew(v)
            mat = np.eye(3) + vx + (1 - c) / (s**2) * vx @ vx;
            R = SO3.from_matrix(mat)

        return R

if __name__ == "__main__":
    R = SO3()