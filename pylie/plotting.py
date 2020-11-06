try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError as error:
    print("The plotting functionality of pylie depends on matplotlib.")
    raise error

from pylie import SO3, SE3, SIM3, LieGroup
import numpy as np

def plotFrame(frame : LieGroup, style='-', ax : Axes3D = None):
    if not isinstance(frame, (SIM3, SE3, SO3)):
        raise TypeError("The frame must be of one of the types: SIM3, SE3, SO3.")
    if ax is None:
        ax = plt.gca()
    if not isinstance(ax, Axes3D):
        raise TypeError("The axes must be of 3D type.")

    if isinstance(frame, SO3):
        mat = np.eye(4)
        mat[0:3,0:3] = frame.as_matrix()
    else:
        mat = frame.as_matrix()
    
    Q = mat[0:3,0:3]
    t = mat[0:3,3:4]

    frame_axes = [[],[],[]]
    for a in range(3):
        for i in range(3):
            frame_axes[a] += [t[a,0], t[a,0]+Q[a,i], np.nan]

    # result = ax.plot(frame_axes[0], frame_axes[1], frame_axes[2])
    colors = ['r','g','b']
    colors = [c+style for c in colors]
    result = []
    for a in range(3):
        temp = ax.plot([t[0,0], t[0,0]+Q[0,a]], [t[1,0], t[1,0]+Q[1,a]], [t[2,0], t[2,0]+Q[2,a]], colors[a])
        result.append(temp)
    temp = ax.plot(t[0,0], t[1,0], t[2,0], 'ko')
    result.append(temp)

    return result

if __name__ == "__main__":
    pose = SE3.identity()
    pose = SE3.from_list([1,2,3], "x")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotFrame(pose)

    plt.show()
