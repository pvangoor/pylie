from .LieGroup import LieGroup
from .SO3 import SO3 as SO3
from .R3 import R3 as R3
import numpy as np

class Quaternion(LieGroup):
    DIM = 4
    def __init__(self, q = None):
        self._quat = np.array((1,0,0,0))
        if isinstance(q, np.ndarray):
            assert q.size == 4, "Quaternions must have 4 components."
            self._quat = q.copy().reshape(4)
        elif isinstance(q, Quaternion):
            self._quat = q._quat
    
    def q(self) -> np.ndarray:
        return self._quat
    
    def real(self) -> float:
        return self._quat[0]
    
    def imag(self) -> np.ndarray:
        return self._quat[1:]
    
    def as_rotation(self):
        r = self._quat[0]
        v = self._quat[1:]
        rot = v.reshape((-1,1)) @ v.reshape((1,-1)) + r**2 * np.eye(3) + SO3.skew(v) @ SO3.skew(v)
        return SO3(rot)
    
    def as_vector(self):
        return self.q().copy()

    def __str__(self):
        return str(self._quat)
    
    def Adjoint(self):
        norm2_q = np.inner(self._quat, self._quat)
        r = self._quat[0]
        u = self._quat[1:]
        Ad = np.zeros((4,4))
        Ad[0,0] = 1.
        Ad[1:,1:] = np.eye(3) +2* (r*SO3.skew(u) + SO3.skew(u)@ SO3.skew(u)) / norm2_q
        return Ad
    
    @staticmethod
    def adjoint(quat_vec : np.ndarray) -> np.ndarray:
        assert isinstance(quat_vec, np.ndarray)
        assert quat_vec.size == 4
        ad = np.zeros((4,4))
        ad[1:,1:] = 2*SO3.skew(quat_vec[1:])
        return ad
    
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            q3 = np.zeros(4)
            q3[0] = self._quat[0]*other._quat[0] - self._quat[1:] @ other._quat[1:]
            q3[1:] = self._quat[0]*other._quat[1:] + other._quat[0]*self._quat[1:] + np.cross(self._quat[1:], other._quat[1:])
            return Quaternion(q3)
        elif isinstance(other, np.ndarray) and other.size == 3:
            other_q = np.zeros(4)
            other_q[1:] = other
            result = self * other_q * self.inv()
            return result[1:]
        elif isinstance(other, np.ndarray) and other.size == 4:
            return self * Quaternion(other)
        elif isinstance(other, R3):
            result = R3()
            result._trans = self * other._trans
            return result
        return NotImplemented
    
    def __truediv__(self, other):
        if isinstance(other, Quaternion):
            return self * other.inv()
        return NotImplemented
    
    def inv(self):
        norm_sq = np.inner(self._quat, self._quat)
        result = self._quat.copy()
        result[1:] = - self._quat[1:]
        result = result / norm_sq
        return Quaternion(result)
    
    def log(self):
        r = self._quat[0]
        u = self._quat[1:]
        norm = np.linalg.norm(self._quat)
        if norm == 0:
            return np.nan * np.empty(4)
        result = np.zeros(4)
        result[0] = np.log(norm)
        norm_u = np.linalg.norm(u)
        if norm_u == 0:
            return result
        result[1:] = np.arctan(norm_u / r) * u / norm_u
        return result
    
    def as_matrix(self):
        q = self._quat
        mat = np.array((
            (q[0], q[1], q[2], q[3]),
            (-q[1], q[0], -q[3], q[2]),
            (-q[2], q[3], q[0], -q[1]),
            (-q[3],-q[2],q[1],q[0])
        ))
        return mat

    @staticmethod
    def from_matrix(mat : np.ndarray) -> 'SO3':
        if not isinstance(mat, np.ndarray):
            raise TypeError
        if not mat.shape == (4,4):
            raise ValueError
        return Quaternion(mat[0,:])
    
    @staticmethod
    def identity():
        return Quaternion(np.array((1,0,0,0)))

    @staticmethod
    def exp(quat_vec):
        assert quat_vec.size == 4, "The so(3) Lie algebra vector must have 4 elements."
        w = quat_vec[1:]
        r = quat_vec[0]
        theta = np.linalg.norm(w)
        q = np.zeros(4)
        if theta == 0:
            q[0] = 1.
        else:
            q[0] = np.cos(theta)
            q[1:] = np.sin(theta)*(w/theta)
        return Quaternion(np.exp(r)*q)

    @staticmethod
    def valid_list_formats() -> dict:
        # Possible formats are
        # q : 4 entry quaternion (scalar last)
        # w : 4 entry quaternion (scalar first)
        return {'q':4, 'w':4}

    @staticmethod
    def from_list(line, format_spec="q") -> 'Quaternion':
        result = Quaternion()
        if format_spec == "q":
            result._quat = np.array([float(line[i]) for i in range(4)])
            line = line[4:]
        elif format_spec == "w":
            result._quat = np.array([float(line[i]) for i in [1,2,3,0]])
            line = line[4:]
        else:
            return NotImplemented
        return result

    def to_list(self, format_spec) -> list:
        if format_spec == "q":
            temp = self._quat.tolist()
            result = [temp[i] for i in [1,2,3,0]]
        elif format_spec == "w":
            result = self._quat.tolist()
        else:
            return NotImplemented
        return result
    
    @staticmethod
    def list_header(format_spec):
        if format_spec == "q":
            result = "qx,qy,qz,qw".split()
        elif format_spec == "w":
            result = "qw,qx,qy,qz".split()
        else:
            return NotImplemented
        return result

    @staticmethod
    def vee(mat : np.ndarray) -> np.ndarray:
        if not isinstance(mat, np.ndarray):
            raise TypeError
        if not mat.shape == (4,4):
            raise ValueError
        return mat[0,:]

    @staticmethod
    def wedge(vec : np.ndarray) -> np.ndarray:
        if not isinstance(vec, np.ndarray):
            raise TypeError
        if not vec.size == 4:
            raise ValueError
        return Quaternion(vec).as_matrix()
    
    