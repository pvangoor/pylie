from abc import ABC, abstractmethod
import numpy as np

class LieGroup(ABC):
    def __str__(self):
        pass

    def __repr__(self):
        return str(self.as_matrix())

    @abstractmethod
    def __mul__(self, other):
        pass
    
    def __matmul__(self, other):
        if isinstance(other, LieGroup):
            return self.as_matrix() @ other.as_matrix()
        elif isinstance(other, np.ndarray):
            return self.as_matrix() @ other
        return NotImplemented
    
    @abstractmethod
    def inv(self):
        pass
    
    def __truediv__(self, other):
        return self * other.inv()
    
    @abstractmethod
    def log(self):
        pass
    
    def Adjoint(self):
        raise NotImplementedError()

    def as_matrix(self):
        raise NotImplementedError()
    
    @staticmethod
    def identity():
        pass

    @staticmethod
    def exp(lie_alg_vector):
        pass

    @staticmethod
    def valid_list_formats():
        return []
    
    @staticmethod
    def from_list(line, format_spec):
        pass
    
    def to_list(self, format_spec):
        raise NotImplementedError()

    @staticmethod
    def list_header(format_spec):
        pass

    @staticmethod
    def vee(mat : np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def wedge(mat : np.ndarray) -> np.ndarray:
        pass