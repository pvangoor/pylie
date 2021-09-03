import pylie
import numpy as np


class Trajectory:
    def __init__(self, elements=None, times=None):
        if elements is None:
            self._elements = []
            self._times = []
        else:
            self._elements = elements

        if times is None:
            self._times = list(range(len(self._elements)))
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
            i0, i1 = self._get_indices_near_time(t)
            if i0 is None:
                return None
            elif i1 is None:
                return self._elements[0]

            # Now (inter/extra)polate
            X0 = self._elements[i0]
            t0 = self._times[i0]
            X1 = self._elements[i1]
            t1 = self._times[i1]

            dt = t1 - t0
            motion = (X0.inv() * X1).log() / dt
            elem = X0 * X0.exp((t - t0) * motion)
            return elem

        elif isinstance(t, list):
            # t is a list of numbers -> return the corresponding trajectory
            return Trajectory([self[tau] for tau in t], t)

        raise NotImplementedError

    def get_velocity(self, t):
        if isinstance(t, (int, float, complex)) and not isinstance(t, bool):
            # t is a number
            t = float(t)
            i0, i1 = self._get_indices_near_time(t)
            if i0 is None:
                return None
            elif i1 is None:
                return self._elements[0].log() * 0.0

            # Now (inter/extra)polate
            X0 = self._elements[i0]
            t0 = self._times[i0]
            X1 = self._elements[i1]
            t1 = self._times[i1]

            motion = (X0.inv() * X1).log() / (t1 - t0)

            assert not np.isnan(motion).any()
            return motion

        raise NotImplementedError

    def get_velocities(self):
        vels = [
            (self._elements[i].inv() * self._elements[i+1]).log() / (self._times[i+1] - self._times[i]) for i in range(len(self)-1)
        ]
        return vels

    def begin_time(self):
        return self._times[0]

    def end_time(self):
        return self._times[-1]

    def get_elements(self):
        return self._elements

    def get_times(self):
        return self._times

    def group_type(self):
        if len(self._elements) > 0:
            return type(self._elements[0])
        else:
            return None

    def __rmul__(self, other):
        if isinstance(other, self.group_type()):
            new_elements = [other * X for X in self._elements]
            new_times = [t for t in self._times]
            return Trajectory(new_elements, new_times)
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, self.group_type()):
            new_elements = [X * other for X in self._elements]
            new_times = [t for t in self._times]
            return Trajectory(new_elements, new_times)
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

        idx1 = [j for j in reversed(
            range(len(self._times))) if self._times[j] <= t1]
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

    def _get_indices_near_time(self, t: float):
        assert(isinstance(t, float))
        if len(self._times) == 0:
            return None, None
        elif len(self._times) == 1:
            return 0, None

        next_idx = [j for j in range(len(self._times)) if self._times[j] > t]
        if len(next_idx) == 0:
            # There are no times after t
            return len(self._times)-2, len(self._times)-1
        elif next_idx[0] == 0:
            # All times are after t
            return 0, 1
        else:
            return next_idx[0]-1, next_idx[0]


if __name__ == "__main__":
    import numpy as np
    times = [i * 0.1 for i in range(10)]
    motion = np.random.randn(3, 1)
    elements = [pylie.SO3.exp(motion*t) for t in times]
    traj = Trajectory(elements, times)

    new_times = [0.2, 0.45, 1.2, -0.12]
    for t in new_times:
        group_error = traj[t] / pylie.SO3.exp(t*motion)
        norm_error = np.linalg.norm(group_error.as_matrix() - np.eye(3))
        print("The error at time {} is {}".format(t, norm_error))
