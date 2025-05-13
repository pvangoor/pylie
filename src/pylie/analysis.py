from .SIM3 import SIM3
import numpy as np
from .Trajectory import Trajectory
from .LieGroup import LieGroup

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
    def _list_to_stack(points):
        if isinstance(points, list):
            return np.stack([p.ravel() for p in points]).T
        else:
            return points
    points1 = _list_to_stack(points1)
    points2 = _list_to_stack(points2)
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
    result._R = result._R.from_matrix(R)
    result._x.__init__(x)
    result._s.__init__(float(s))

    return result


def RKMK3_integrate(X0 : LieGroup, f, h : float) -> LieGroup:
    # X0 is the initial condition
    # f is the vector field in the form $\dot{X} = X f(X)$.
    # h is the step size
    # Based on Algorithm 3.2 from the paper:
    # Munthe-Kaas, Hans. "Runge-Kutta methods on Lie groups." BIT Numerical Mathematics 38 (1998): 92-111.

    # Set up the coefficients
    A = np.array([
        [0,0,0],
        [0.5,0,0],
        [-1.,2.,0],
    ])
    b = np.array([1/6.,2/3.,1/6.])
    # c = np.array([0,0.5,1.0])

    # Compute the iteration
    I1 = f(X0)
    k = [I1]
    for i in range(1,3):
        u = h * A[i,:i] @ k
        k.append(f(X0 * X0.exp(u)))
        
    v = h * sum(b[j] * k[j] for j in range(3))
    vTilde = v + h / 6. * X0.adjoint(I1) @ v
    X1 = X0 * X0.exp(vTilde)

    return X1


def RKMK4_integrate(X0 : LieGroup, f, h : float) -> LieGroup:
    # X0 is the initial condition
    # f is the vector field in the form $\dot{X} = X f(X)$.
    # h is the step size
    # Based on Algorithm 3.3 from the paper:
    # Munthe-Kaas, Hans. "Runge-Kutta methods on Lie groups." BIT Numerical Mathematics 38 (1998): 92-111.

    # Set up the coefficients
    A = np.array([
        [0,0,0,0],
        [0.5,0,0,0],
        [0,0.5,0,0],
        [0,0,1.0,0],
    ])
    b = np.array([1/6.,1/3.,1/3.,1/6.])
    c = np.array([0,0.5,0.5,1.0])
    d = A @ c
    coeff_matrix = np.array([
        [c[1], c[1]**2, 2*d[1]],
        [c[2], c[2]**2, 2*d[2]],
        [c[3], c[3]**2, 2*d[3]],
    ])
    m = np.linalg.solve(coeff_matrix.T, np.array([1.,0,0]))

    # Compute the iteration
    I1 = f(X0)
    k = [I1]
    for i in range(1,4):
        u = h * A[i,:i] @ k
        uTilde = u + c[i]*h/6. * X0.adjoint(I1) @ u
        k.append(f(X0 * X0.exp(uTilde)))
        
    I2 = sum(m[i] * (k[i+1] - I1) for i in range(3)) / h
    v = h * sum(b[j] * k[j] for j in range(4))
    vTilde = v + h / 4. * X0.adjoint(I1) @ v + h**2 / 24. * X0.adjoint(I2) @ v
    X1 = X0 * X0.exp(vTilde)

    return X1
