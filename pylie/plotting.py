try:
    import matplotlib.pyplot as plt
    from matplotlib.artist import Artist
    from matplotlib.patches import Wedge, Circle, Arc
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

        if isinstance(pose_data, SO3):
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
    
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

class ArtificialHorizonArtist(Artist):
    def __init__(self, ax : plt.Axes, attitude_data : SO3):
        if not isinstance(ax, plt.Axes):
            raise TypeError("The axes must be of plt.Axes type.")
        if not isinstance(attitude_data, SO3):
            raise TypeError("The attitude_data must be an SO3 element.")

        # Background
        ah_circle = Circle((0,0), radius=1.0, color='w', ec='w')
        ah_colouring = Circle((0,0), radius=5.0, color='k')
        ax.add_patch(ah_colouring)
        ax.add_patch(ah_circle)


        # Horizon line
        self._attitude_data = attitude_data
        true_horizon = self._compute_horizon(attitude_data)
        self._horizon_line, = ax.plot(true_horizon[0,:], true_horizon[1,:])

        # Horizon shading
        height = self._compute_horizon_height(attitude_data)
        slope = self._compute_horizon_slope(attitude_data) * 180.0 / np.pi
        self._shade = Wedge((0,height), 2.0, -180+slope, slope, alpha=0.5)
        ax.add_patch(self._shade)

        # Clip path
        clip_circle = Circle((0, 0), radius=1.0, transform=ax.transData)
        self._horizon_line.set_clip_path(clip_circle)
        self._shade.set_clip_path(clip_circle)

        # Center Overlay
        ax.add_patch(Arc((0,0), 0.3, 0.3, theta1=180, lw=2.0))
        ax.plot([0.15, 0.6], [0,0], color='k', lw=2.0)
        ax.plot([-0.15, -0.6], [0,0], color='k', lw=2.0)

        # Axes settings
        ax.axis('square')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
    
    def set_attitude_data(self, attitude : SE3):
        if not isinstance(attitude, SO3):
            raise TypeError("The attitude_data must be an SO3 element.")
        
        self._attitude_data = attitude
        true_horizon = self._compute_horizon(attitude)
        self._horizon_line.set_data(true_horizon[0,:], true_horizon[1,:])

        height = self._compute_horizon_height(attitude)
        slope = self._compute_horizon_slope(attitude) * 180.0 / np.pi
        self._shade.set_center((0,height))
        self._shade.set_theta1(-180 + slope)
        self._shade.set_theta2(slope)
    
    @staticmethod
    def _compute_horizon(attitude : SO3):
        angles = np.linspace(0, 2*np.pi, 50)
        base_horizon = np.vstack((np.cos(angles), np.sin(angles), np.zeros(angles.shape)))
        true_horizon = attitude.inv() * base_horizon
        true_horizon = true_horizon[:, true_horizon[0,:]>0]
        true_horizon = true_horizon / true_horizon[0,:]
        true_horizon = true_horizon[1:,:]
        return true_horizon
    
    @staticmethod
    def _compute_horizon_slope(attitude : SO3):
        theta = ArtificialHorizonArtist._find_heading_angle(attitude)
        H = attitude.inv() * np.array([[np.cos(theta)], [np.sin(theta)], [0.0]])
        DH = attitude.inv() * np.array([[-np.sin(theta)], [np.cos(theta)], [0.0]])
        DH2 = H[0,0] * DH - DH[0,0] * H
        slope = np.arctan2(DH2[2,0], DH2[1,0])
        return slope
    
    @staticmethod
    def _compute_horizon_height(attitude : SO3):
        theta = ArtificialHorizonArtist._find_heading_angle(attitude)
        H = attitude.inv() * np.array([[np.cos(theta)], [np.sin(theta)], [0.0]])
        H = H / H[0,0]
        return H[2,0]
    
    @staticmethod
    def _find_heading_angle(attitude):
        RT = attitude.as_matrix().T
        theta = np.arctan2(-RT[1,0], RT[1,1])
        return theta




def plotFrame(frame : LieGroup, style='-', ax : Axes3D = None):
    if ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot(111, projection='3d')
    
    frame_artist = frameArtist(ax, frame, style)

    return frame_artist

def plotArtificialHorizon(attitude : SO3, ax : Axes3D = None):
    if ax is None:
        ax = plt.gca()
    horizon_artist = ArtificialHorizonArtist(ax, attitude)
    return horizon_artist

if __name__ == "__main__":
    pose = SE3.identity()
    pose = SE3.from_list([1,2,3], "x")
    pose2 = SE3.from_list([4,5,6], "x")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    artist = plotFrame(pose)
    plt.pause(1)

    artist.set_pose_data(pose2)
    plt.show()

    plt.cla()

    yaw_tf = SO3.from_list(np.array([0,0,0.5]), 'r')
    att = SO3.from_list(np.array([0.1,0,0.]), 'r')
    att2 = SO3.from_list(np.array([0.2,0,0]), 'r')
    for i in range(10):
        plt.cla()
        att = SO3.from_list(np.array([0.1*i,0.0,0]), 'r')
        artist = plotArtificialHorizon(att)
        plt.pause(0.2)

    for i in range(10):
        plt.cla()
        att = SO3.from_list(np.array([0.0,-0.05*i,0]), 'r')
        artist = plotArtificialHorizon(att)
        plt.pause(0.2)
    
    plt.show()

