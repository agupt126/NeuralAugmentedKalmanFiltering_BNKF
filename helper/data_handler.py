import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class TrajectoryReader:
    def __init__(self, traj_file_name, relative_dataset_path = '../dataset', sigma_pos=1, sigma_vel=1e-3):
        self.traj_file_name = traj_file_name
        self.trajectory_df = None
        self.sigma_pos = sigma_pos
        self.sigma_vel = sigma_vel

        if traj_file_name:
            self.relative_dataset_path = relative_dataset_path
            self.__read_traj()
            self.traj_name = self.traj_file_name.split(".")[0]
            self.gif_path = f'{relative_dataset_path}/GIFS/{self.traj_name}_animation.gif'
            self.gif_path_measured = f'{relative_dataset_path}/GIFS/{self.traj_name}_MEASURED_animation.gif'

            self.add_velocity_feature()
            self.add_delta_time()
            #self.add_noise_measurements()
            #self.add_noise_measurements_t_plus_1()

    def __read_traj(self):
        self.trajectory_df = pd.read_csv(f'{self.relative_dataset_path}/Synthetic-UAV-Flight-Trajectories/{self.traj_file_name}',
                                         dtype={'timestamp': 'float64', 'tx': 'float64', 'ty': 'float64',
                                                'tz': 'float64'})
        self.trajectory_df['timestamp'] = self.trajectory_df['timestamp'] - self.trajectory_df['timestamp'].iloc[0]
        self.trajectory_df['timestamp'] = self.trajectory_df['timestamp'].astype('float32')
        return self.trajectory_df

    def add_velocity_feature(self, sigma=None):
        self.trajectory_df['vx'] = self.trajectory_df['tx'].diff().shift(-1) / self.trajectory_df['timestamp'].diff().shift(-1)
        self.trajectory_df['vy'] = self.trajectory_df['ty'].diff().shift(-1) / self.trajectory_df['timestamp'].diff().shift(-1)
        self.trajectory_df['vz'] = self.trajectory_df['tz'].diff().shift(-1) / self.trajectory_df['timestamp'].diff().shift(-1)

        self.trajectory_df.fillna({'vx': 0}, inplace=True)
        self.trajectory_df.fillna({'vy': 0}, inplace=True)
        self.trajectory_df.fillna({'vz': 0}, inplace=True)
        return None

    def add_noise_measurements(self):
        num_values = len(self.trajectory_df['timestamp'])
        self.trajectory_df['tx_measured'] = self.trajectory_df['tx'] + np.random.normal(0, self.sigma_pos, size=num_values)
        self.trajectory_df['ty_measured'] = self.trajectory_df['ty'] + np.random.normal(0, self.sigma_pos, size=num_values)
        self.trajectory_df['tz_measured'] = self.trajectory_df['tz'] + np.random.normal(0, self.sigma_pos, size=num_values)

        self.trajectory_df['vx_measured'] = self.trajectory_df['vx'] + np.random.normal(0, self.sigma_vel, size=num_values)
        self.trajectory_df['vy_measured'] = self.trajectory_df['vy'] + np.random.normal(0, self.sigma_vel, size=num_values)
        self.trajectory_df['vz_measured'] = self.trajectory_df['vz'] + np.random.normal(0, self.sigma_vel, size=num_values)

        self.trajectory_df['sigma_pos'] = self.sigma_pos
        self.trajectory_df['sigma_vel'] = self.sigma_vel
        return None

    def add_noise_measurements_t_plus_1(self):
        self.trajectory_df['tx_measured_plus'] = self.trajectory_df['tx_measured'].shift(-1, fill_value=0)
        self.trajectory_df['ty_measured_plus'] = self.trajectory_df['ty_measured'].shift(-1, fill_value=0)
        self.trajectory_df['tz_measured_plus'] = self.trajectory_df['tz_measured'].shift(-1, fill_value=0)
        return None

    def add_delta_time(self):
        self.trajectory_df['delta_time'] = self.trajectory_df['timestamp'].diff().shift(-1)
        self.trajectory_df.fillna({'delta_time': 0}, inplace=True)
        return None

    ## visualizations
    def create_traj_scatter(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.trajectory_df['tx'], self.trajectory_df['ty'], self.trajectory_df['tz'], color='b', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()
        return None

    def create_animation(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        line, = ax.plot([], [], [], lw=2)

        ax.set_xlim(self.trajectory_df['tx'].min(), self.trajectory_df['tx'].max())
        ax.set_ylim(self.trajectory_df['ty'].min(), self.trajectory_df['ty'].max())
        ax.set_zlim(self.trajectory_df['tz'].min(), self.trajectory_df['tz'].max())

        ax.set_xlabel('X Pos')
        ax.set_ylabel('Y Pos')
        ax.set_zlabel('Z Pos')

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            return line,

        def update(frame):
            line.set_data(self.trajectory_df['tx'][:frame], self.trajectory_df['ty'][:frame])
            line.set_3d_properties(self.trajectory_df['tz'][:frame])
            return line,

        ani = FuncAnimation(fig, update, frames=len(self.trajectory_df), init_func=init, blit=True, interval=100)
        ani.save(self.gif_path, writer='pillow', fps=30)

        plt.show()
        return None


    def create_traj_scatter_measured(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.trajectory_df['tx_measured'], self.trajectory_df['ty_measured'], self.trajectory_df['tz_measured'], color='b', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()
        return None

    def create_animation_measured(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        line, = ax.plot([], [], [], lw=2)

        ax.set_xlim(self.trajectory_df['tx_measured'].min(), self.trajectory_df['tx_measured'].max())
        ax.set_ylim(self.trajectory_df['ty_measured'].min(), self.trajectory_df['ty_measured'].max())
        ax.set_zlim(self.trajectory_df['tz_measured'].min(), self.trajectory_df['tz_measured'].max())

        ax.set_xlabel('X Pos')
        ax.set_ylabel('Y Pos')
        ax.set_zlabel('Z Pos')
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            return line,
        def update(frame):
            line.set_data(self.trajectory_df['tx_measured'][:frame], self.trajectory_df['ty_measured'][:frame])
            line.set_3d_properties(self.trajectory_df['tz_measured'][:frame])
            return line,

        ani = FuncAnimation(fig, update, frames=len(self.trajectory_df), init_func=init, blit=True, interval=100)
        ani.save(self.gif_path_measured, writer='pillow', fps=30)

        plt.show()
        return None