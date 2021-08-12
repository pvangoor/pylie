from .SIM3 import SIM3
import numpy as np
from .Trajectory import Trajectory

def align_trajectory(trajectory0 : Trajectory, trajectory1 : Trajectory, ret_params=False) -> Trajectory:
    # Align trajectory0 to trajectory1
    assert isinstance(trajectory0, Trajectory)
    assert isinstance(trajectory1, Trajectory)
    
    t0 = max(trajectory0.begin_time(), trajectory1.begin_time())
    t1 = min(trajectory0.end_time(), trajectory1.end_time())
    if t0 >= t1:
        if ret_params:
            return Trajectory(), SIM3.identity()
        else:
            return Trajectory()
    
    num_points = 100
    times = np.linspace(t0, t1, num_points).tolist()
    points0 = [trajectory0[t].x().as_vector() for t in times]
    points1 = [trajectory1[t].x().as_vector() for t in times]

    S = umeyama(points0, points1)

    trajectoryA = S.to_SE3() * trajectory0
    if ret_params:
        return trajectoryA, S
    else:
        return trajectoryA



def umeyama(points1 : np.ndarray, points2 : np.ndarray) -> SIM3:
    # This function solves the least squares problem of finding a SIM3 transform S such that
    # S * points1 = points2,
    # s_S * R_S * points1_i + x_S = points2_i
    if isinstance(points1, list):
        points1 = np.hstack(points1)
    if isinstance(points2, list):
        points2 = np.hstack(points2)
    assert isinstance(points1, np.ndarray) and isinstance(points2, np.ndarray), "The points must be in numpy arrays."
    assert points1.shape == points2.shape, "The points are not matched."
    assert points1.shape[0] == 3, "The points are not 3D."

    # Compute relevant variables
    n = points1.shape[1]
    mu1 = 1/n * np.reshape(np.sum(points1, axis=1),(3,-1))
    mu2 = 1/n * np.reshape(np.sum(points2, axis=1),(3,-1))
    sig1sq = 1/n * np.sum((points1 - mu1)**2.0)
    sig2sq = 1/n * np.sum((points2 - mu2)**2.0)
    Sig12 = 1/n * (points2-mu2) @ (points1-mu1).T

    # Use the SVD for the rotation
    U, d, Vh = np.linalg.svd(Sig12)
    S = np.eye(3)
    if np.linalg.det(Sig12) < 0:
        S[-1,-1] = -1
    
    R = U @ S @ Vh
    s = 1.0 / sig1sq * np.trace(np.diag(d) @ S)
    x = mu2 - s * R @ mu1

    # Return the result as a SIM3 element
    result = SIM3()
    result._R._rot = result._R._rot.from_matrix(R)
    result._x._trans = x
    result._s._scale = float(s)

    return result

if __name__ == "__main__":
    true_transform = SIM3.from_list(np.random.randn(1,7).ravel().tolist(), "srx")
    print(true_transform)

    points1 = np.random.randn(3,10)
    points2 = true_transform * points1

    est_transform = umeyama(points1, points2)
    print(est_transform)
    