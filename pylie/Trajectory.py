import pylie
import numpy as np

class Trajectory:
    def __init__(self, elements = None, times = None):
        if elements is None:
            self._elements = []
            self._times = []
        else:
            self._elements = elements

        if times is None:
            self._times = list(range(len(elements)))
        else:
            self._times = times
        
        self._clean_duplicates()
    
    def __len__(self):
        assert len(self._elements) == len(self._times)
        return len(self._elements)

    def __getitem__(self, t):
        if isinstance(t, (int, float, complex)) and not isinstance(t, bool):
            # t is a number
            t = float(t)
            next_idx = [j for j in range(len(self._times)) if self._times[j] > t]
            if len(next_idx) == 0:
                next_idx = len(self._times)-1
            else:
                next_idx = next_idx[0]
            if next_idx == 0:
                next_idx = 1
            
            # Now (inter/extra)polate
            base_element = self._elements[next_idx-1]
            dt = self._times[next_idx] - self._times[next_idx-1]
            motion = (base_element.inv() * self._elements[next_idx]).log() / dt
            ndt = t - self._times[next_idx-1]
            elem = base_element * base_element.exp(ndt * motion)
            return elem

        raise NotImplementedError

    def get_velocity(self, t):
        if isinstance(t, (int, float, complex)) and not isinstance(t, bool):
            # t is a number
            t = float(t)
            next_idx = [j for j in range(len(self._times)) if self._times[j] > t]
            if len(next_idx) == 0:
                next_idx = len(self._times)-1
            else:
                next_idx = next_idx[0]
            if next_idx == 0:
                next_idx = 1
            
            # Now (inter/extra)polate
            dt = self._times[next_idx] - self._times[next_idx-1]
            motion = (self._elements[next_idx-1].inv() * self._elements[next_idx]).log() / dt

            assert not np.isnan(motion).any()
            return motion

        raise NotImplementedError

    def begin_time(self):
        return self._times[0]
    
    def end_time(self):
        return self._times[-1]
    
    def get_elements(self):
        return self._elements
    
    def group_type(self):
        if len(self._elements) > 0:
            return type(self._elements[0])
        else:
            return None
    
    def __rmul__(self, other):
        if isinstance(other, self.group_type()):
            new_elements = [other * X for X in self._elements]
            new_times = [t for t in self._times]
            return Trajectory(new_elements,new_times)
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, self.group_type()):
            self._elements = [X * other for X in self._elements]
            new_times = [t for t in self._times]
            return Trajectory(new_elements,new_times)
        raise NotImplementedError
    
    def truncate(self, t0, t1):
        assert t0 < t1

        idx0 = [j for j in range(len(self._times)) if self._times[j] >= t0]
        if len(idx0) > 0:
            idx0 = idx0[0]
        else:
            self._times.clear()
            self._elements.clear()
            return

        idx1 = [j for j in reversed(range(len(self._times))) if self._times[j] <= t1]
        if len(idx1) > 0:
            idx1 = idx1[0]
        else:
            self._times.clear()
            self._elements.clear()
            return
        
        self._times = self._times[idx0:idx1]
        self._elements = self._elements[idx0:idx1]
    
    def _clean_duplicates(self):
        for i in reversed(range(len(self._times)-1)):
            if self._times[i+1] - self._times[i] < 1e-8:
                del self._times[i+1]
                del self._elements[i+1]


if __name__ == "__main__":
    import numpy as np
    times = [i * 0.1 for i in range(10)]
    motion = np.random.randn(3,1)
    elements = [pylie.SO3.exp(motion*t) for t in times]
    traj = Trajectory(elements, times)

    new_times = [0.2, 0.45, 1.2, -0.12]
    for t in new_times:
        group_error = traj[t] / pylie.SO3.exp(t*motion)
        norm_error = np.linalg.norm(group_error.as_matrix() - np.eye(3))
        print("The error at time {} is {}".format(t, norm_error))
        
