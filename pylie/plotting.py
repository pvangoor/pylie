try:
    import matplotlib.pyplot as plt
    from matplotlib.artist import Artist
    from mpl_toolkits.mplot3d import Axes3D
except ImportError as error:
    print("The plotting functionality of pylie depends on matplotlib.")
    raise error

from pylie import SO3, SE3, SIM3, LieGroup
import numpy as np

class frameArtist(Artist):
    def __init__(self, ax : Axes3D, pose_data : SE3, style='-'):
        if not isinstance(ax, Axes3D):
            raise TypeError("The axes must be of 3D type.")
        if not isinstance(pose_data, (SIM3, SE3, SO3)):
            raise TypeError("The pose_data must be of one of the types: SIM3, SE3, SO3.")

        self._pose_data = pose_data
        Q, t = self._pose_to_Qt(pose_data)

        colors = ['r','g','b']
        colors = [c+style for c in colors]
        self._frame_lines = []
        for a in range(3):
            temp, = ax.plot([t[0,0], t[0,0]+Q[0,a]], [t[1,0], t[1,0]+Q[1,a]], [t[2,0], t[2,0]+Q[2,a]], colors[a])
            self._frame_lines.append(temp)
        self._frame_center, = ax.plot(t[0,0], t[1,0], t[2,0], 'ko')
    
    def set_pose_data(self, pose : SE3):
        if not isinstance(pose, (SIM3, SE3, SO3)):
            raise TypeError("The frame must be of one of the types: SIM3, SE3, SO3.")
        
        self._pose_data = pose
        Q, t = self._pose_to_Qt(pose)
        for a in range(3):
            self._frame_lines[a].set_data_3d([t[0,0], t[0,0]+Q[0,a]], [t[1,0], t[1,0]+Q[1,a]], [t[2,0], t[2,0]+Q[2,a]])
        self._frame_center.set_data_3d(t[0,0], t[1,0], t[2,0])

    
    @staticmethod
    def _pose_to_Qt(pose : SE3):
        if isinstance(pose, SO3):
            mat = np.eye(4)
            mat[0:3,0:3] = pose.as_matrix()
        else:
            mat = pose.as_matrix()
        
        Q = mat[0:3,0:3]
        t = mat[0:3,3:4]
        return Q, t


def plotFrame(frame : LieGroup, style='-', ax : Axes3D = None):
    if ax is None:
        ax = plt.gca()
    
    frame_artist = frameArtist(ax, frame, style)

    return frame_artist

if __name__ == "__main__":
    pose = SE3.identity()
    pose = SE3.from_list([1,2,3], "x")
    pose2 = SE3.from_list([4,5,6], "x")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    artist = plotFrame(pose)
    plt.show()

    artist.set_pose_data(pose2)
    plt.show()
